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

from aif_gen.api.prompt_mapper import PromptMapper
from aif_gen.api.response_mapper import ResponseMapper
from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.generate.caching import AsyncElasticsearchCache
from aif_gen.task.alignment_task import AlignmentTask


def get_judge_model_prompt(
    prompt: str, chosen: str, rejected: str, preference: str
) -> str:
    return (
        'Please evaluate whether "response 1" is more aligned with the "prompt", compared to "response 2" according to the following preference.'
        f'Preference: {preference}\n'
        'Respond with "1" if "response 1" is more aligned, and "0" if "response 2" is more aligned.'
        f'Prompt: {prompt}\n\n'
        f'Response 1: {chosen}\n\n'
        f'Response 2: {rejected}\n\n'
        'Alignment (1 or 0):'
    )


async def generate_continual_dataset(
    data_config: Dict[str, Any],  # TODO: Should bind this type
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    max_tokens_prompt_response: int = 1024,
    max_tokens_chosen_rejected_response: int = 2048,
    dry_run: bool = False,
    include_preference_axes: bool = False,
    temperature: float = 1.0,
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

    Returns:
        Optional[ContinualAlignmentDataset]: The synthetically generated dataset.
    """
    prompt_mapper = PromptMapper()
    response_mapper = ResponseMapper()

    task_specs = data_config['task_specs']

    if dry_run:
        logging.info(f'Doing dry-run data generation on a single sample...')
        mock_task = AlignmentTask.from_dict(task_specs[0]['alignment_task'])
        cache = await AsyncElasticsearchCache.maybe_from_env_var(
            index_name=f'CACHE_DATA_GENERATION_{model_name}'
        )
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
            cache=cache,
            temperature=temperature,
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(
                f'Exception occurred on dry-run, skipping generation: {e}'
            )
            raise e
        finally:
            if cache is not None:
                await cache.close()

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
            if include_preference_axes:
                coro = _generate_sample_with_preference_axes(
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

        # need to use the judge model if the preference axes are included
        # will be used to ensure chosen and reject are consistent with the task preference
        if include_preference_axes:
            cache_judge = await AsyncElasticsearchCache.maybe_from_env_var(
                index_name=f'CACHE_DATA_GENERATION_JUDGE_{model_name}'
            )
            assert isinstance(continual_dataset, ContinualAlignmentDataset)

            futures = []
            datasets = continual_dataset.datasets
            for dataset_idx, dataset in enumerate(datasets):
                dataset_size = len(dataset)
                logging.info(
                    f'Judging preference on Dataset ({dataset_size} samples) {dataset.task}'
                )
                dataset_preference = dataset.task.preference
                for sample in dataset.samples:
                    prompt = sample.prompt
                    chosen = sample.chosen
                    rejected = sample.rejected
                    from aif_gen.dataset.validation.llm_judge import _get_score

                    alignment_coro = _get_score(
                        get_judge_model_prompt(
                            prompt, chosen, rejected, dataset_preference
                        ),
                        client,
                        model_name,
                        async_semaphore,
                        64,
                        dataset_idx=dataset_idx,
                        metric_name='alignment_generation',
                        cache=cache_judge,
                    )

                    futures.append(asyncio.create_task(alignment_coro))  # type: ignore

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
                        logging.warning(
                            f'Dataset {dataset_idx} is empty, skipping judging preference.'
                        )
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
                        sample_score = scores[sample_idx]
                        if sample_score not in (0.0, 1.0):
                            logging.warning(
                                f'Bad judge score {sample_score!r} for sample {sample_idx}, skipping.'
                            )
                            continue
                        if sample_score == 0.0:
                            sample.chosen, sample.rejected = (
                                sample.rejected,
                                sample.chosen,
                            )
                            # swap if judge says response 2 is better

                logging.info('Judging preference completed.')

            except BaseException as e:
                logging.exception(f'Exception occurred while judging preference: {e}')
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
def get_tries(default: int = 3) -> int:
    if 'BACKOFF_RETRIES' in os.environ:
        try:
            return int(os.environ['BACKOFF_RETRIES'])
        except:
            logging.warning(f'Failed to parse BACKOFF_RETRIES, using: {default}')
    return default


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=get_tries(),
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
    cache: 'AsyncElasticsearchCache | None' = None,
    temperature: float = 1.0,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Generate a AlignmentDataset dataset given the AlignmentTask, and model.

    Args:
        task (AlignmentTask): The AlignmentTask to generate data for.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        prompt_mapper (PromptMapper): Creates the 'meta-prompt' for this sample's task prompt.
        response_mapper (ResponseMapper): Created the 'meta-prompt' for this sample's response prompt.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_prompt_response (int): Configurable limit on the max_tokens for the generated prompt response.
        max_tokens_chosen_rejected_response (int): Configurable limit on the max_tokens for the generated chosen and rejected response.
        dataset_idx (int): The idx of the dataset that the sample is requested for to align out-of-order asyn execution.
        max_tokens (int): Max number of tokens to generate.
        prompt_idx (int): The idx of the sample, to distinguish between multiple requests for the same task.
        cache (AsyncElasticsearchCache): Optionally specify a AsyncElasticsearchCache instance for caching.
        temperature (float): Temperature for the model.

    Returns:
        Optional[Tuple[AlignmentDatasetSample, int]]: A single sample of the dataset, and the dataset idx (None if pydantic.ValidationError occurred).

    Raises:
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    try:

        class _PromptProposal(pydantic.BaseModel, extra='forbid'):
            prompt: str

        class _ResponsePair(pydantic.BaseModel, extra='forbid'):
            chosen: str
            rejected: str

        meta_prompt = prompt_mapper.generate_prompt(task)
        meta_prompt_nonce = f'{prompt_idx}'

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(meta_prompt, nonce=meta_prompt_nonce)
            else:
                output = None

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
                assert output is not None  # This is for mypy

        if output is None:
            raise ValueError(f'Received None response to prompt: {meta_prompt}')

        prompt = _PromptProposal.model_validate_json(output).prompt

        # Update/set cache only after validating output JSON.
        if cache is not None:
            await cache.set(query=meta_prompt, value=output, nonce=meta_prompt_nonce)

        task_prompt = response_mapper.generate_prompt(task, prompt)
        logging.debug(
            f'Meta Prompt: {meta_prompt} (Nonce: {meta_prompt_nonce}), '
            f'Model Response: {prompt}'
        )

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(task_prompt)
            else:
                output = None

            if output is None:
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
                assert output is not None  # This is for mypy

        if prompt is None:
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


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=get_tries(),
)
async def _generate_sample_with_preference_axes(
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
    cache: 'AsyncElasticsearchCache | None' = None,
    temperature: float = 1.0,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Generate a AlignmentDataset dataset given the AlignmentTask, and model including the preference axes during response mapping.

    Args:
        task (AlignmentTask): The AlignmentTask to generate data for.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        prompt_mapper (PromptMapper): Creates the 'meta-prompt' for this sample's task prompt.
        response_mapper (ResponseMapper): Created the 'meta-prompt' for this sample's response prompt.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_prompt_response (int): Configurable limit on the max_tokens for the generated prompt response.
        max_tokens_chosen_rejected_response (int): Configurable limit on the max_tokens for the generated chosen and rejected response.
        dataset_idx (int): The idx of the dataset that the sample is requested for to align out-of-order asyn execution.
        max_tokens (int): Max number of tokens to generate.
        prompt_idx (int): The idx of the sample, to distinguish between multiple requests for the same task.
        cache (AsyncElasticsearchCache): Optionally specify a AsyncElasticsearchCache instance for caching.
        temperature (float): Temperature for the model.

    Returns:
        Optional[Tuple[AlignmentDatasetSample, int]]: A single sample of the dataset, and the dataset idx (None if pydantic.ValidationError occurred).

    Raises:
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    try:

        class _PromptProposal(pydantic.BaseModel, extra='forbid'):
            prompt: str

        class _Response(pydantic.BaseModel, extra='forbid'):
            response: str

        class ResponsePair(pydantic.BaseModel, extra='forbid'):
            chosen: str  # no more the semantics of chosen - just for naming
            rejected: str  # no more the semantics of rejected - just for naming

        meta_prompt = prompt_mapper.generate_prompt(task)
        meta_prompt_nonce = f'{prompt_idx}'

        async with async_semaphore:
            if cache is not None:
                output: Optional[str] = await cache.get(
                    meta_prompt, nonce=meta_prompt_nonce
                )
            else:
                output = None

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
                assert output is not None  # This is for mypy

        if output is None:
            raise ValueError(f'Received None response to prompt: {meta_prompt}')

        prompt = _PromptProposal.model_validate_json(output).prompt

        # Update/set cache only after validating output JSON.
        if cache is not None:
            await cache.set(query=meta_prompt, value=output, nonce=meta_prompt_nonce)

        # generate a list of randomly generated scores each between 1 and 5
        scores = [
            random.randint(1, 5)
            for _ in range(response_mapper.NUMBER_OF_PREFERENCE_AXES_SAMPLED)
        ]
        task_prompt = response_mapper.generate_no_preference_prompt(
            task,
            prompt,
            scores,
            parity=0,
        )
        task_prompt_second = response_mapper.generate_no_preference_prompt(
            task,
            prompt,
            scores,
            parity=1,
        )
        logging.debug(
            f'Meta Prompt: {meta_prompt} (Nonce: {meta_prompt_nonce}), '
            f'Model Response: {prompt}'
        )

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(task_prompt + task_prompt_second)
                if output is None:
                    raise ValueError(
                        f'No cached response for task prompt: {task_prompt + task_prompt_second}'
                    )
                structured_output = ResponsePair.model_validate_json(output)
                output1_str: str = structured_output.chosen
                output2_str: str = structured_output.rejected
            else:
                output = None

            if output is None:
                task1 = asyncio.create_task(
                    client.chat.completions.create(
                        model=model_name,
                        messages=[{'role': 'user', 'content': task_prompt}],
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
                )
                task2 = asyncio.create_task(
                    client.chat.completions.create(
                        model=model_name,
                        messages=[{'role': 'user', 'content': task_prompt_second}],
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
                )
                resp1 = await task1
                resp2 = await task2
                [resp1, resp2]
                output1 = resp1.choices[0].message.content
                output2 = resp2.choices[0].message.content
                # guard against None, then narrow the types
                if output1 is None or output2 is None:
                    raise ValueError(
                        f'Expected two JSON strings, got: {output1!r}, {output2!r}'
                    )
                output1_str = output1
                output2_str = output2

        if prompt is None:
            raise ValueError(f'Received None response to prompt: {prompt}')

        structured_response1 = _Response.model_validate_json(output1_str)
        structured_response2 = _Response.model_validate_json(output2_str)
        if (
            structured_response1.response is None
            or structured_response2.response is None
        ):
            raise ValueError(
                f'Received None response to prompt: {prompt}, '
                f'Output1: {output1_str}, Output2: {output2_str}'
            )
        combined_response: dict[str, str] = {
            'chosen': structured_response1.response,
            'rejected': structured_response2.response,
        }
        output = json.dumps(combined_response)
        structured_response = ResponsePair.model_validate_json(output)
        if cache is not None:
            await cache.set(query=(task_prompt + task_prompt_second), value=output)

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


async def filter_continual_alignment_dataset_style_normalize(
    input_dataset: ContinualAlignmentDataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    max_tokens: int = 4096,
    dry_run: bool = False,
    temperature: float = 1.0,
) -> Optional[ContinualAlignmentDataset]:
    r"""Normalize the style between chosen/rejected responses while preserving quality differences.

    Args:
        input_dataset (ContinualAlignmentDataset): Dataset to filter and normalize.
        model_name (str): The model alias to use for generating normalized responses.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens (int): Configurable limit on the max_tokens for the generated responses.
        dry_run (bool): If True, process a single sample to ensure the model is setup correctly.
        temperature (float): Temperature for the model.

    Returns:
        Optional[ContinualAlignmentDataset]: The filtered and normalized dataset.
    """
    if dry_run:
        logging.info(f'Doing dry-run style normalization on a single sample...')
        mock_task = input_dataset.datasets[0].task
        mock_sample = input_dataset[0]
        cache = await AsyncElasticsearchCache.maybe_from_env_var(
            index_name=f'CACHE_STYLE_NORMALIZATION_{model_name}'
        )
        coro = _normalize_sample_style(
            mock_task,
            mock_sample,  # type: ignore
            client,
            model_name,
            async_semaphore,
            max_tokens,
            dataset_idx=-1,
            cache=cache,
            temperature=temperature,
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(
                f'Exception occurred on dry-run, skipping normalization: {e}'
            )
            raise e
        finally:
            if cache is not None:
                await cache.close()

        logging.info('Dry run was a success.')
        return None

    cache = await AsyncElasticsearchCache.maybe_from_env_var(
        index_name=f'CACHE_STYLE_NORMALIZATION_{model_name}'
    )
    futures, tasks, dataset_sizes = [], [], []
    for dataset_idx, dataset in enumerate(input_dataset.datasets):
        task = dataset.task
        dataset_size = len(dataset)
        logging.info(f'Style normalizing Dataset ({dataset_size} samples) {task}')

        tasks.append(task)
        dataset_sizes.append(dataset_size)
        for _sample_idx in range(dataset_size):
            coro = _normalize_sample_style(
                task,
                dataset[_sample_idx],  # type: ignore
                client,
                model_name,
                async_semaphore,
                max_tokens,
                dataset_idx=dataset_idx,
                cache=cache,
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
            train_frac = input_dataset.datasets[i].train_frac
            continual_dataset.append(AlignmentDataset(tasks[i], samples[i], train_frac))
        return continual_dataset
    except BaseException as e:
        logging.exception(f'Exception occurred while normalizing dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None
    finally:
        if cache is not None:
            await cache.close()


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError),
    max_tries=get_tries(),
)
async def _normalize_sample_style(
    task: AlignmentTask,
    sample: AlignmentDatasetSample,
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
    max_tokens: int,
    dataset_idx: int,
    cache: 'AsyncElasticsearchCache | None' = None,
    temperature: float = 1.0,
) -> Optional[Tuple[AlignmentDatasetSample, int]]:
    r"""Normalize the style between chosen/rejected responses in a single sample.

    Args:
        task (AlignmentTask): The AlignmentTask for context.
        sample (AlignmentDatasetSample): The sample to normalize.
        client (openai.AsyncOpenAI): Handle to openAI client.
        model_name (str): openAI model alias.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens (int): Max number of tokens to generate.
        dataset_idx (int): The idx of the dataset that the sample is from.
        cache (AsyncElasticsearchCache): Optionally specify a cache instance for caching.
        temperature (float): Temperature for the model.

    Returns:
        Optional[Tuple[AlignmentDatasetSample, int]]: The normalized sample and dataset index.

    Raises:
        openai.NotFoundError: If the openAI model cannot be accessed at the configured endpoint.
    """
    try:

        class _ResponsePair(pydantic.BaseModel, extra='forbid'):
            chosen: str
            rejected: str

        prompt = (
            'Given the following RLHF data point (chosen, reject), rewrite both so that they do not differ in style '
            '(e.g., punctuation, tone), but preserve the difference in quality and the key idea in each. '
            "Return the 'chosen' and 'rejected' fields.\n\n"
            f'Response 1 (chosen):\n{sample.chosen}\n\nResponse 2 (rejected):\n{sample.rejected}'
        )

        cache_key = f'style_norm_{str(task)}_{hash(sample.chosen + sample.rejected)}'

        async with async_semaphore:
            if cache is not None:
                output = await cache.get(cache_key)
            else:
                output = None

            if output is None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
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
                assert output is not None  # This is for mypy

        if output is None:
            raise ValueError(f'Received None response to prompt: {prompt}')

        structured_response = _ResponsePair.model_validate_json(output)
        if cache is not None:
            await cache.set(query=cache_key, value=output)

        # Update the sample with normalized responses
        sample.chosen = structured_response.chosen
        sample.rejected = structured_response.rejected

        logging.debug(f'Normalized sample: {sample}')
        return sample, dataset_idx
    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None
