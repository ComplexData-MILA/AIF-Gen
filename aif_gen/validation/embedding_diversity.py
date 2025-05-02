import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar

import backoff
import numpy as np
import openai
from tqdm.asyncio import tqdm

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


async def llm_embedding_diversity(
    dataset: Dataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    batch_size: int,
    async_semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> Optional[List[Optional[Dict[str, float]]]]:
    r"""Use the cosine distance of embeddings from an embedding model as a proxy for dataset diversity.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        model_name (str): The vLLM-compatible model alias to use for embedding texts.
        client (openai.AsyncOpenAI): Handle to openAI client.
        batch_size (int): Number of items to submit at a time.
        async_semaphore (asyncio.Semaphore): Semaphore that manages number of concurrent API requests.
        dry_run (bool): If True, validate a dummy sample to ensure the model is setup correctly.

    Returns:
        Optional[List[Optional[Dict[str, float]]]]: For every AlignmentDataset, returns a dictionary with entries of the form '{metric}_{stat}':
            - Stat is one of ['mean', 'median', 'min', 'max']
            - Metric denotes the embedding diversity for either ['prompt', 'chosen', 'rejected'] tokens.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.
    """
    if dry_run:
        logging.info(f'Doing dry-run data validation on a single sample...')
        coro = _batch_embed(
            [''],
            client=client,
            model_name=model_name,
            async_semaphore=async_semaphore,
            extra_data='prompt',
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(f'Exception on dry-run, skipping validation: {e}')
            raise e
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

        for batch in _batch_iterable(dataset.samples, batch_size=batch_size):
            texts = {
                'prompt': [sample.prompt for sample in batch],
                'chosen': [sample.chosen for sample in batch],
                'rejected': [sample.rejected for sample in batch],
            }
            for text_type, text in texts.items():
                coro = _batch_embed(
                    text,
                    client=client,
                    model_name=model_name,
                    async_semaphore=async_semaphore,
                    extra_data={'dataset_idx': dataset_idx, 'text_type': text_type},
                )
                futures.append(asyncio.create_task(coro))
    try:
        results: List[Dict[str, List]] = [
            defaultdict(list) for _ in range(len(datasets))
        ]
        for fut in tqdm.as_completed(futures, total=len(futures)):
            result = await fut
            if result is None:
                continue
            embeddings, extra_data = result
            dataset_idx, text_type = extra_data['dataset_idx'], extra_data['text_type']
            results[dataset_idx][text_type].extend(embeddings)

        aggregated_results: List[Optional[Dict[str, float]]] = []
        for i, dataset in enumerate(datasets):
            if not len(dataset):
                logging.warning('Skipping Embedding Diversity Eval for empty dataset')
                aggregated_results.append(None)
                continue

            for metric_name, metric_values in results[i].items():
                if len(metric_values) != len(dataset):
                    logging.warning(
                        f'Dataset {i} {metric_name} validation coverage: {len(metric_values)} / {len(dataset)}'
                    )
                if len(metric_values) == 0:
                    raise RuntimeError(f'No samples could be parsed in dataset {i}')

            aggregated_results.append(_compute_statistics(results[i]))
        return aggregated_results

    except BaseException as e:
        logging.exception(f'Exception occurred while generating dataset: {e}')
        for fut in futures:
            fut.cancel()
        await tqdm.gather(*futures)
        return None


T = TypeVar('T')


def _batch_iterable(
    iterable: Iterable[T], batch_size: int
) -> Generator[List[T], None, None]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@backoff.on_exception(backoff.expo, (openai.RateLimitError,))
async def _batch_embed(
    texts: list[str],
    client: openai.AsyncOpenAI,
    model_name: str,
    async_semaphore: asyncio.Semaphore,
    extra_data: Any,
) -> tuple[List[List[float]], Any]:
    async with async_semaphore:
        response = await client.embeddings.create(
            input=texts, model=model_name, encoding_format='float'
        )
        embeddings: List[List[float]] = [_data.embedding for _data in response.data]
    return embeddings, extra_data


def _avg_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-10  # avoid division by zero

    normalized_matrix = matrix / row_norms  # shape: (n, m)
    sum_normalized = normalized_matrix.sum(axis=0)  # shape: (m,)
    return normalized_matrix.dot(sum_normalized) / matrix.shape[0]


def _compute_statistics(results: Dict[str, List[List[float]]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        embeddings = np.asarray(values)  # (dataset, embed_dim)
        similarity = _avg_cosine_similarity(embeddings)  # (dataset,)
        diversity: np.ndarray = 1 - similarity

        statistics[f'{metric}_mean'] = float(np.mean(diversity))
        statistics[f'{metric}_median'] = float(np.median(diversity))
        statistics[f'{metric}_min'] = float(np.min(diversity))
        statistics[f'{metric}_max'] = float(np.max(diversity))
    return statistics
