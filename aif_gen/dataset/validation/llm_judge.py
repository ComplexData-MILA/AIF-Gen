import asyncio
import logging
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import backoff
import numpy as np
import openai
import pydantic
from tqdm.asyncio import tqdm

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.generate.caching import AsyncElasticsearchCache
from aif_gen.typing import Dataset


async def llm_judge_validation(
    dataset: Dataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    async_semaphore: asyncio.Semaphore,
    max_tokens_judge_response: int = 32,
    dry_run: bool = False,
) -> Optional[List[Optional[Dict[str, float]]]]:
    r"""Use an LLM to judge the quality of the dataset..

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        model_name (str): The vLLM-compatible model alias to use for validating the data.
        client (openai.AsyncOpenAI): Handle to openAI client.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        max_tokens_judge_response (int): Configurable limit on the max_tokens for the generated judge response.
        dry_run (bool): If True, validate a dummy sample to ensure the model is setup correctly.

    Returns:
        Optional[List[Optional[Dict[str, float]]]]: For every AlignmentDataset, returns a dictionary with entries of the form '{metric}_{stat}':
            - Stat is one of ['mean', 'median', 'min', 'max']
            - Metric is one of:
                'alignment'           -> Whether the chosen response is more aligned with the prompt compared to the rejected response.
                'coherence_chosen'    -> The coherence in the chosen response, as determined by the LLM.
                'coherence_rejected'  -> The coherence in the rejected response, as determined by the LLM.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.
    """
    cache = await AsyncElasticsearchCache.maybe_from_env_var(
        f'CACHE_VALIDATION_{model_name}'
    )

    if dry_run:
        logging.info(f'Doing dry-run data validation on a single sample...')
        mock_sample = AlignmentDatasetSample('Mock', 'Mock', 'Mock')
        _prompt = _get_alignment_prompt(
            mock_sample.prompt, mock_sample.chosen, mock_sample.rejected
        )
        coro = _get_score(
            _prompt,
            client,
            model_name,
            async_semaphore,
            max_tokens_judge_response,
            dataset_idx=-1,
            metric_name='',
            cache=cache,
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(f'Exception occured on dry-run, skipping validation: {e}')
            raise e
        finally:
            if cache is not None:
                await cache.close()

        logging.info('Dry run was a success.')
        return None

    if isinstance(dataset, AlignmentDataset):
        datasets = [dataset]
    else:
        # This assert is here to make mypy happy
        assert isinstance(dataset, ContinualAlignmentDataset)
        datasets = dataset.datasets

    futures = []
    for dataset_idx, dataset in enumerate(datasets):
        dataset_size = len(dataset)
        logging.info(f'Validating Dataset ({dataset_size} samples)')

        for sample in dataset.samples:
            alignment_coro = _get_score(
                _get_alignment_prompt(sample.prompt, sample.chosen, sample.rejected),
                client,
                model_name,
                async_semaphore,
                max_tokens_judge_response,
                dataset_idx=dataset_idx,
                metric_name='alignment',
                cache=cache,
            )
            coherence_chosen_coro = _get_score(
                _get_coherence_prompt(sample.chosen),
                client,
                model_name,
                async_semaphore,
                max_tokens_judge_response,
                dataset_idx=dataset_idx,
                metric_name='coherence_chosen',
                cache=cache,
            )
            coherence_rejected_coro = _get_score(
                _get_coherence_prompt(sample.rejected),
                client,
                model_name,
                async_semaphore,
                max_tokens_judge_response,
                dataset_idx=dataset_idx,
                metric_name='coherence_rejected',
                cache=cache,
            )
            futures.append(asyncio.create_task(alignment_coro))
            futures.append(asyncio.create_task(coherence_chosen_coro))
            futures.append(asyncio.create_task(coherence_rejected_coro))

    try:
        results: List[Dict[str, List[float]]] = [defaultdict(list)] * len(datasets)
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is None:
                continue

            score, dataset_idx, metric_name = result
            if score is not None:
                results[dataset_idx][metric_name].append(score)

        aggregated_results: List[Optional[Dict[str, float]]] = []
        for i, dataset in enumerate(datasets):
            if not len(dataset):
                logging.warning('Skipping LLM judge for empty dataset')
                aggregated_results.append(None)
                continue

            for metric_name, metric_values in results[i].items():
                if len(metric_values) != len(dataset):
                    logging.warning(
                        f'Dataset {i} {metric_name} validation coverage: {len(metric_values)} / {len(dataset)}'
                    )
                if len(metric_values) == 0:
                    raise RuntimeError(
                        f'Could not parse LLM output for any samples in dataset {i}'
                    )
            aggregated_results.append(_compute_statistics(results[i]))
        return aggregated_results

    except BaseException as e:
        logging.exception(f'Exception occured while generating dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures, return_exceptions=True)
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


@backoff.on_exception(backoff.expo, (openai.RateLimitError,), max_tries=get_tries())
async def _get_score(
    prompt: str,
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
    max_tokens_judge_response: int,
    dataset_idx: int,
    metric_name: str,
    cache: Optional[AsyncElasticsearchCache] = None,
) -> Tuple[Optional[float], int, str]:
    try:

        class _ValidationResponse(pydantic.BaseModel, extra='forbid'):  # type: ignore
            score: float

        async with async_semaphore:
            model_response: Optional[str] = None
            if cache is not None:
                model_response = await cache.get(prompt)

            if model_response is None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens_judge_response,
                    response_format={
                        'type': 'json_schema',
                        'json_schema': {
                            'name': 'SyntheticPreference',
                            'schema': _ValidationResponse.model_json_schema(),
                            'strict': True,
                        },
                    },
                )
                model_response = response.choices[0].message.content
                if model_response is None:
                    raise ValueError(f'Received None response to prompt: {prompt}')
                assert model_response is not None  # This is for mypy

        score = _ValidationResponse.model_validate_json(model_response).score

        # Write to cache only if JSON is valid.
        if cache:
            await cache.set(query=prompt, value=model_response)

        score = max(0, min(1, score))
        logging.debug(f'Prompt: {prompt}, Response: {model_response}, Score: {score}')
        return score, dataset_idx, metric_name

    except pydantic.ValidationError as e:
        logging.error(f'Failed to bind structured output json schema: {e}')
        return None, dataset_idx, metric_name


def _get_alignment_prompt(prompt: str, chosen: str, rejected: str) -> str:
    return (
        'Please evaluate whether "response 1" is more aligned with the "prompt", compared to "response 2". '
        'Consider alignment with the prompt in terms of clarity, relevance, and correctness regarding preferences specified in the prompt. '
        'Respond with "1" if "response 1" is more aligned, and "0" if "response 2" is more aligned.'
        f'Prompt: {prompt}\n\n'
        f'Response 1: {chosen}\n\n'
        f'Response 2: {rejected}\n\n'
        'Alignment (0 or 1):'
    )


def _get_coherence_prompt(response: str) -> str:
    return (
        'Please evaluate the coherence of the following response on a scale from 0 to 1, '
        'where 1 indicates excellent coherence and 0 indicates poor coherence:\n\n'
        f'Response: {response}\n\n'
        'Coherence Score (0 to 1):'
    )


def _compute_statistics(results: Dict[str, List[float]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        statistics[f'{metric}_mean'] = float(np.mean(values))
        statistics[f'{metric}_median'] = float(np.median(values))
        statistics[f'{metric}_min'] = float(np.min(values))
        statistics[f'{metric}_max'] = float(np.max(values))
    return statistics
