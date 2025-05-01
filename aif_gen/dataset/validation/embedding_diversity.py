import asyncio
import logging
from collections import defaultdict
from typing import Dict, Generator, Iterable, List, Optional, TypeVar

import backoff
import numpy as np
import openai
from tqdm.asyncio import tqdm

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.typing import Dataset


async def llm_embedding_diversity(
    dataset: Dataset,
    model_name: str,
    client: openai.AsyncOpenAI,
    batch_size: int,
    async_semaphore: asyncio.Semaphore,
    dry_run: bool = False,
) -> Optional[List[Optional[Dict[str, float]]]]:
    r"""Use an LLM to judge the quality of the dataset..

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
            - Metric is one of:
                'alignment'           -> Whether the chosen response is more aligned with the prompt compared to the rejected response.
                'coherence_chosen'    -> The coherence in the chosen response, as determined by the LLM.
                'coherence_rejected'  -> The coherence in the rejected response, as determined by the LLM.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.
    """
    if dry_run:
        logging.info(f'Doing dry-run data validation on a single sample...')
        mock_sample = AlignmentDatasetSample('Mock', 'Mock', 'Mock')
        coro = _batch_embed(
            [mock_sample.prompt],
            client=client,
            model_name=model_name,
            async_semaphore=async_semaphore,
            extra_data='prompt',
        )
        try:
            _ = await coro
        except BaseException as e:
            logging.exception(
                f'Exception occurred on dry-run, skipping validation: {e}'
            )
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
            prompts = [sample.prompt for sample in batch]
            chosen = [sample.chosen for sample in batch]
            rejected = [sample.rejected for sample in batch]

            embed_prompt_coro = _batch_embed(
                prompts,
                client=client,
                model_name=model_name,
                async_semaphore=async_semaphore,
                extra_data={'dataset_idx': dataset_idx, 'text_type': 'prompt'},
            )
            embed_chosen_coro = _batch_embed(
                chosen,
                client=client,
                model_name=model_name,
                async_semaphore=async_semaphore,
                extra_data={'dataset_idx': dataset_idx, 'text_type': 'chosen'},
            )
            embed_rejected_coro = _batch_embed(
                rejected,
                client=client,
                model_name=model_name,
                async_semaphore=async_semaphore,
                extra_data={'dataset_idx': dataset_idx, 'text_type': 'rejected'},
            )
            futures.append(asyncio.create_task(embed_prompt_coro))
            futures.append(asyncio.create_task(embed_chosen_coro))
            futures.append(asyncio.create_task(embed_rejected_coro))

    try:
        results: List[Dict[str, List[List[float]]]] = [defaultdict(list)] * len(
            datasets
        )
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
                    raise RuntimeError(
                        f'Could not parse LLM output for any samples in dataset {i}'
                    )

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
    """Splits an iterable into batches of specified size.

    Args:
        iterable: The input iterable to be batched.
        batch_size: The maximum number of items per batch.

    Yields:
        Lists of items, each up to `batch_size` in length.
    """
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
    extra_data: 'T | None' = None,
) -> tuple[List[List[float]], 'T | None']:
    """Embed text and return extra_data verbatim."""
    async with async_semaphore:
        response = await client.embeddings.create(
            input=texts,
            model=model_name,
            encoding_format='float',
        )
        embeddings: List[List[float]] = [_data.embedding for _data in response.data]
        assert isinstance(embeddings[0][0], float)  # should not be base64-encoded.

    return embeddings, extra_data


def _cosine_similarity_matrix_self_transpose(matrix: np.ndarray) -> np.ndarray:
    """Return the average cosine similarity for each row with all rows of the matrix.

    This function normalizes each row and then computes the average cosine similarity.
    It avoids constructing the full n x n similarity matrix, which is memory-efficient
    when the number of rows (n) is much larger than the number of columns (m).

    Args:
        matrix (np.ndarray): A 2D NumPy array of shape (n, m).

    Returns:
        np.ndarray: A 1D array of shape (n,) containing the average cosine similarity for each row.
    """
    # Compute the norm of each row and avoid division by zero
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-10  # avoid division by zero

    # Normalize the matrix rows
    normalized_matrix = matrix / row_norms  # shape: (n, m)

    # Compute the sum of all normalized rows
    sum_normalized = normalized_matrix.sum(axis=0)  # shape: (m,)
    n = matrix.shape[0]

    # Compute average cosine similarity for each row
    avg_similarity = normalized_matrix.dot(sum_normalized) / n
    return avg_similarity


def _compute_statistics(results: Dict[str, List[List[float]]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        embeddings = np.asarray(values)  # (dataset, embed_dim)
        similarity = _cosine_similarity_matrix_self_transpose(embeddings)  # (dataset,)
        diversity: np.ndarray = 1 - similarity

        statistics[f'{metric}_mean'] = float(np.mean(diversity))
        statistics[f'{metric}_median'] = float(np.median(diversity))
        statistics[f'{metric}_min'] = float(np.min(diversity))
        statistics[f'{metric}_max'] = float(np.max(diversity))
    return statistics
