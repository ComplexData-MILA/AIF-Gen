import asyncio
import json
import logging
import os
import random
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import backoff
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.generate.caching import AsyncElasticsearchCache
from aif_gen.generate.mappers import PromptMapper, ResponseMapper
from aif_gen.generate.mappers.pair_selector import ScoredCandidate, select_pair
from aif_gen.task.alignment_task import AlignmentTask


async def generate_continual_dataset(
    data_config: Dict[str, Any],
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    max_tokens_prompt_response: int = 1024,
    max_tokens_chosen_rejected_response: int = 2048,
    dry_run: bool = False,
    include_preference_axes: bool = False,
    temperature: float = 1.0,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Optional[ContinualAlignmentDataset]:
    r"""Generate a ContinualAlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        data_config (Dict[str, Any]): Configuration file storing tasks specifications and model info.
        model_name (str): The vLLM-compatible model alias to use for generation synthetic samples.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_prompt_response (int): Configurable limit on the max_tokens for the generated prompt response.
        max_tokens_chosen_rejected_response (int): Configurable limit on the max_tokens for the generated chosen and rejected response.
        dry_run (bool): If True, ignore the config and generate a dummy sample to ensure the model is setup correctly.
        include_preference_axes (bool): If True, include the preference axes in the prompt for response mapper.
        temperature (float): Temperature for the model.
        generation_config (Optional[Dict[str, Any]]): Optional dict enabling the difficulty tuned pipeline.
            When provided, overrides the legacy joint-generation path. Recognized
            keys: ``difficulty`` (int in [1,4]), ``prob`` (float in [0, 1]),

    Returns:
        Optional[ContinualAlignmentDataset]: The synthetically generated dataset.
    """
    prompt_mapper = PromptMapper()
    response_mapper = ResponseMapper()
    task_specs = data_config['task_specs']
    use_difficulty_pipeline = generation_config is not None
    if use_difficulty_pipeline:
        assert generation_config is not None  # for type narrowing
        logging.info(
            f'Using difficulty-tuned generation pipeline: {generation_config}'
        )

    if dry_run:
        logging.info(f'Doing dry-run data generation on a single sample...')
        mock_task = AlignmentTask.from_dict(task_specs[0]['alignment_task'])
        if use_difficulty_pipeline:
            coro = _generate_difficulty_sample(
                mock_task,
                client,
                model_name,
                prompt_mapper,
                response_mapper,
                async_semaphore,
                max_tokens_prompt_response,
                max_tokens_chosen_rejected_response,
                dataset_idx=-1,
                prompt_idx=-1,
                cache=None,
                generation_config=generation_config,
                temperature=temperature,
            )
        else:
            coro = _generate_sample(
                mock_task,
                client,
                model_name,
                prompt_mapper,
                response_mapper,
                async_semaphore,
                max_tokens_prompt_response,
                max_tokens_chosen_rejected_response,
                dataset_idx=-1,
                prompt_idx=-1,
                cache=None,
                include_preference_axes=include_preference_axes,
                temperature=temperature,
            )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(f'Exception on dry-run, skipping generation: {e}')
            raise e
        logging.info('Dry run was a success.')
        return None

    cache = await AsyncElasticsearchCache.maybe_from_env_var(
        index_name=f'CACHE_DATA_GENERATION_{model_name}'
    )
    futures, tasks, dataset_sizes = [], [], []
    for dataset_idx, task_spec in enumerate(task_specs):
        task = AlignmentTask.from_dict(task_spec['alignment_task'])
        dataset_size = task_spec['num_samples']
        logging.info(f'Generating Dataset ({dataset_size} samples) {task}')

        tasks.append(task)
        dataset_sizes.append(dataset_size)
        for _sample_idx in range(dataset_size):
            if use_difficulty_pipeline:
                coro = _generate_difficulty_sample(
                    task,
                    client,
                    model_name,
                    prompt_mapper,
                    response_mapper,
                    async_semaphore,
                    max_tokens_prompt_response,
                    max_tokens_chosen_rejected_response,
                    dataset_idx=dataset_idx,
                    prompt_idx=_sample_idx,
                    cache=cache,
                    generation_config=generation_config,
                    temperature=temperature,
                )
            else:
                coro = _generate_sample(
                    task,
                    client,
                    model_name,
                    prompt_mapper,
                    response_mapper,
                    async_semaphore,
                    max_tokens_prompt_response,
                    max_tokens_chosen_rejected_response,
                    dataset_idx=dataset_idx,
                    prompt_idx=_sample_idx,
                    cache=cache,
                    include_preference_axes=include_preference_axes,
                    temperature=temperature,
                )
            futures.append(asyncio.create_task(coro))

    try:
        samples: List[List[AlignmentDatasetSample]] = [
            [] for _ in range(len(dataset_sizes))
        ]
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is not None:
                sample, dataset_idx = result
                samples[dataset_idx].append(sample)

        continual_dataset = ContinualAlignmentDataset(datasets=[])
        for i in range(len(samples)):
            if len(samples[i]) != dataset_sizes[i]:
                logging.warning(
                    f'Dataset {i} requested {dataset_sizes[i]} samples but LM generated {len(samples[i])}'
                )
            continual_dataset.append(AlignmentDataset(tasks[i], samples[i]))

        # If preference axes included, use judge to pick chosen/rejected responses.
        # Skipped under the sampled pipeline, which already orders the pair by rubric score.
        if include_preference_axes and not use_difficulty_pipeline:
            from aif_gen.validation.llm_judge import (
                _get_judge_prompt,
                _get_score,
            )

            cache_judge = await AsyncElasticsearchCache.maybe_from_env_var(
                index_name=f'CACHE_DATA_GENERATION_JUDGE_{model_name}'
            )
            assert isinstance(continual_dataset, ContinualAlignmentDataset)

            futures = []
            datasets = continual_dataset.datasets
            for dataset_idx, dataset in enumerate(datasets):
                dataset_size = len(dataset)
                logging.info(f'Judging dataset ({dataset_size} samples) {dataset.task}')
                preference = dataset.task.preference
                for sample in dataset.samples:
                    judge_coro = _get_score(
                        _get_judge_prompt(
                            sample.prompt, sample.chosen, sample.rejected, preference
                        ),
                        client,
                        model_name,
                        async_semaphore,
                        max_tokens_judge_response=64,
                        dataset_idx=dataset_idx,
                        metric_name='alignment_generation',
                        cache=cache_judge,
                    )
                    futures.append(asyncio.create_task(judge_coro))  # type: ignore
            try:
                results: List[Dict[str, List[float]]] = [
                    defaultdict(list) for _ in range(len(datasets))
                ]
                for fut in tqdm.as_completed(futures, total=len(futures)):
                    result = await fut
                    if result is None:
                        continue

                    score, dataset_idx, metric_name = result
                    if score is not None:
                        results[dataset_idx][metric_name].append(score)

                for dataset_idx, dataset in enumerate(datasets):
                    if not len(dataset):
                        logging.warning(f'Dataset {dataset_idx} empty, skipping judge.')
                        continue

                    dataset_scores = results[dataset_idx]
                    dataset_samples = dataset.samples
                    for sample_idx, sample in enumerate(dataset_samples):
                        # guard against missing / malformed scores
                        scores = dataset_scores.get('alignment_generation', [])
                        if sample_idx >= len(scores):
                            logging.warning(
                                f'No judge score for sample {sample_idx} in dataset {dataset_idx}, skipping.'
                            )
                            continue
                        score = scores[sample_idx]
                        if score not in (0.0, 1.0):
                            logging.warning(
                                f'Bad judge score {score!r} for sample {sample_idx}, skipping.'
                            )
                            continue
                        if score == 0.0:  # swap if judge says response 2 is better
                            sample.chosen, sample.rejected = (
                                sample.rejected,
                                sample.chosen,
                            )
                logging.info('Judging preference completed.')

            except BaseException as e:
                logging.exception(f'Exception while judging preference: {e}')
                for fut in futures:
                    fut.cancel()
                await asyncio.gather(*futures, return_exceptions=True)
                return None
            finally:
                if cache_judge is not None:
                    await cache_judge.close()
                if cache is not None:
                    await cache.close()
        return continual_dataset

    except BaseException as e:
        logging.exception(f'Exception occurred while generating dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None

    finally:
        if cache is not None:
            await cache.close()


@lru_cache(maxsize=None)
def _get_tries(default: int = 3) -> int:
    if 'BACKOFF_RETRIES' in os.environ:
        try:
            return int(os.environ['BACKOFF_RETRIES'])
        except:
            logging.warning(f'Failed to parse BACKOFF_RETRIES, using: {default}')
    return default


class _PromptProposal(pydantic.BaseModel, extra='forbid'):
    prompt: str


class _Response(pydantic.BaseModel, extra='forbid'):
    response: str


class _ResponsePair(pydantic.BaseModel, extra='forbid'):
    chosen: str
    rejected: str


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=_get_tries(),
)
async def _generate_sample(
    task: AlignmentTask,
    client: openai.AsyncOpenAI,
    model_name: str,
    prompt_mapper: PromptMapper,
    response_mapper: ResponseMapper,
    async_semaphore: asyncio.Semaphore,
    max_tokens_prompt_response: int,
    max_tokens_chosen_rejected_response: int,
    dataset_idx: int,
    prompt_idx: int,
    include_preference_axes: bool,
    cache: 'AsyncElasticsearchCache | None' = None,
    temperature: float = 1.0,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    try:
        meta_prompt = prompt_mapper.generate_prompt(task)
        meta_prompt_nonce = f'{prompt_idx}'

        async with async_semaphore:
            output = None
            if cache is not None:
                output = await cache.get(meta_prompt, nonce=meta_prompt_nonce)

            if output is None:  # Cache miss or no cache - make API call
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': meta_prompt}],
                    max_tokens=max_tokens_prompt_response,
                    response_format={
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'PromptProposal',
                            'schema': _PromptProposal.model_json_schema(),
                            'strict': True,
                        },
                    },
                    temperature=temperature,
                )
                output = response.choices[0].message.content

        if output is None:
            raise ValueError(f'Received None response to prompt: {meta_prompt}')
        prompt = _PromptProposal.model_validate_json(output).prompt
        if cache is not None:
            await cache.set(query=meta_prompt, value=output, nonce=meta_prompt_nonce)
        logging.debug(
            f'Meta Prompt: {meta_prompt} (Nonce: {meta_prompt_nonce}), Model Response: {prompt}'
        )

        if include_preference_axes:
            task_prompt1, task_prompt2 = response_mapper.generate_no_preference_prompt(
                task, prompt
            )
            task_prompt = task_prompt1 + task_prompt2

            async with async_semaphore:
                output = None
                if cache is not None:
                    output = await cache.get(task_prompt1 + task_prompt2)

                if output is None:  # Cache miss or no cache - make API calls
                    futures = []
                    for response_prompt in [task_prompt1, task_prompt2]:
                        coro = client.chat.completions.create(
                            model=model_name,
                            messages=[{'role': 'user', 'content': response_prompt}],
                            max_tokens=max_tokens_chosen_rejected_response,
                            response_format={
                                'type': 'json_schema',
                                'json_schema': {
                                    'name': 'SyntheticPreference',
                                    'schema': _Response.model_json_schema(),
                                    'strict': True,
                                },
                            },
                            temperature=temperature,
                        )
                        futures.append(asyncio.create_task(coro))

                    resp1 = await futures[0]
                    resp2 = await futures[1]
                    output1_str = resp1.choices[0].message.content
                    output2_str = resp2.choices[0].message.content
                    struct_resp = _Response.model_validate_json(output1_str)  # type: ignore
                    struct_resp2 = _Response.model_validate_json(output2_str)  # type: ignore
                    if struct_resp.response is None or struct_resp2.response is None:
                        raise ValueError(
                            f'Received None response to prompt: {prompt}, Output1: {output1_str}, Output2: {output2_str}'
                        )
                    output = json.dumps(
                        {
                            'chosen': struct_resp.response,
                            'rejected': struct_resp2.response,
                        }
                    )
        else:
            task_prompt = response_mapper.generate_prompt(task, prompt)
            async with async_semaphore:
                output = None
                if cache is not None:
                    output = await cache.get(task_prompt)

                if output is None:  # Cache miss or no cache - make API call
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[{'role': 'user', 'content': task_prompt}],
                        max_tokens=max_tokens_chosen_rejected_response,
                        response_format={
                            'type': 'json_schema',
                            'json_schema': {
                                'name': 'SyntheticPreference',
                                'schema': _ResponsePair.model_json_schema(),
                                'strict': True,
                            },
                        },
                        temperature=temperature,
                    )
                    output = response.choices[0].message.content

        if output is None:
            raise ValueError(f'Received None response to prompt: {prompt}')
        structured_response = _ResponsePair.model_validate_json(output)
        if cache is not None:
            await cache.set(query=task_prompt, value=output)

        sample = AlignmentDatasetSample(
            prompt,
            chosen=structured_response.chosen,
            rejected=structured_response.rejected,
        )
        logging.debug(f'Task Prompt: {task_prompt}, Sample: {sample}')
        return sample, dataset_idx
    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None


# ---------------------------------------------------------------------------
# difficulty pipeline
# ---------------------------------------------------------------------------


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=_get_tries(),
)
async def _generate_difficulty_sample(
    task: AlignmentTask,
    client: openai.AsyncOpenAI,
    model_name: str,
    prompt_mapper: PromptMapper,
    response_mapper: ResponseMapper,
    async_semaphore: asyncio.Semaphore,
    max_tokens_prompt_response: int,
    max_tokens_chosen_rejected_response: int,
    dataset_idx: int,
    prompt_idx: int,
    generation_config: Optional[Dict[str, Any]],
    cache: 'AsyncElasticsearchCache | None' = None,
    temperature: float = 1.0,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Generate one sample.
    1. Generate the task prompt (same as legacy).
    2. Sample difficulty level.
    3. Sample one response persona pair configuration that matches difficulty level.
    4. Generate chosen and rejected responses based on that persona pair configuration.
    """

    difficulty = int(generation_config['difficulty'])
    prob = float(generation_config['prob'])

    try:
        # ---- 1. Generate the task prompt (cached, same as legacy) ----
        meta_prompt = prompt_mapper.generate_prompt(task)
        meta_prompt_nonce = f'{prompt_idx}'

        async with async_semaphore:
            output = None
            if cache is not None:
                output = await cache.get(meta_prompt, nonce=meta_prompt_nonce)
            if output is None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': meta_prompt}],
                    max_tokens=max_tokens_prompt_response,
                    response_format={
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'PromptProposal',
                            'schema': _PromptProposal.model_json_schema(),
                            'strict': True,
                        },
                    },
                    temperature=temperature,
                )
                output = response.choices[0].message.content

        if output is None:
            raise ValueError(f'Received None response to meta prompt: {meta_prompt}')
        prompt = _PromptProposal.model_validate_json(output).prompt
        if cache is not None:
            await cache.set(query=meta_prompt, value=output, nonce=meta_prompt_nonce)

        # ---- 2. Sample a difficulty level ----
        if random.random() < prob:
            D = difficulty
        else:
            D = random.choice([x for x in [1,2,3,4] if x != difficulty])
        
        # ---- 3. Sample a (chosen_persona, rejected_persona) pair consistent with D ----
        chosen_persona, rejected_persona = response_mapper.get_persona_pair(D)

        # ---- 4. Sample chosen and rejected responses in parallel ----
        async def _one_response(
            persona: str, t: float
        ) -> Optional[str]:
            cand_prompt = response_mapper.generate_persona_prompt(
                task, prompt, persona
            )
            cand_nonce = f'{prompt_idx}:{persona}:{t:.3f}'
            async with async_semaphore:
                cand_output = None
                if cache is not None:
                    cand_output = await cache.get(cand_prompt, nonce=cand_nonce)
                if cand_output is None:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=[{'role': 'user', 'content': cand_prompt}],
                        max_tokens=max_tokens_chosen_rejected_response,
                        response_format={
                            'type': 'json_schema',
                            'json_schema': {
                                'name': 'CandidateResponse',
                                'schema': _Response.model_json_schema(),
                                'strict': True,
                            },
                        },
                        temperature=t,
                    )
                    cand_output = resp.choices[0].message.content
            if cand_output is None:
                return None
            try:
                parsed = _Response.model_validate_json(cand_output)
            except pydantic.ValidationError as e:
                logging.warning(f'Persona {persona} schema parse failed: {e}')
                return None
            if cache is not None:
                await cache.set(query=cand_prompt, value=cand_output, nonce=cand_nonce)
            return parsed.response

        chosen_response = _one_response(chosen_persona, temperature)
        rejected_response = _one_response(rejected_persona, temperature)
        
        cr_texts = await asyncio.gather(*[chosen_response, rejected_response], return_exceptions=False)
        for text in cr_texts:
            if text is None:
                return None

        sample = AlignmentDatasetSample(
            prompt=prompt,
            chosen=cr_texts[0],
            rejected=cr_texts[1]
        )

        logging.debug(
            f'Sampled pipeline: prompt_idx={prompt_idx} '
            f'persona_pair=({chosen_persona}, {rejected_persona}) '
        )
        return sample, dataset_idx

    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None
