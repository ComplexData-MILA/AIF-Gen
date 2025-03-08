import logging
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional

import nltk
import numpy as np
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset


def diversity_validation(
    dataset: Dataset, num_workers: int, ngram: int = 3, n_references: Optional[int] = 20
) -> List[Optional[Dict[str, float]]]:
    r"""Report the inverse Self-BLEU score as a measure of diversity within the generated samples.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        num_workers (int): Number of sub-process workers to spawn for the diversity validation.
        ngram (int): The maximum n-gram order for BLEU calculation. Default of 3 matches the original paper.
        n_references (Optional[int]): The number of references to sample for each BLEU calculation. Uses all data samples if None.

    Returns:
        List[Optional[Dict[str, float]]]: For every AlignmentDataset, returns a dictionary with entries of the form '{metric}_{stat}':
            - Stat is one of ['mean', 'median', 'min', 'max']
            - Metric is one of:
                'prompt_diversity'    -> The diversity across prompts in samples of the AlignmentDataset.
                'chosen_diversity'    -> The diversity across chosen responses in samples of the AlignmentDataset.
                'rejected_diversity'  -> The diversity across rejected responses in samples of the AlignmentDataset.

    Note:
        - If the dataset is empty, we put None in place of the dictionary.

    References:
        - https://arxiv.org/pdf/1802.01886
    """
    if not (isinstance(ngram, int) and ngram > 0):
        raise ValueError(f'ngram must be a positive integer, got: {ngram}')

    _download_nltk_resources()

    if isinstance(dataset, AlignmentDataset):
        datasets = [dataset]
    else:
        # This assert is here to make mypy happy
        assert isinstance(dataset, ContinualAlignmentDataset)
        datasets = dataset.datasets

    results: List[Optional[Dict[str, float]]] = []
    for dataset in datasets:
        if len(dataset):
            result = _diversity_validation(dataset, num_workers, ngram, n_references)
        else:
            logging.warning(f'Skipping diversity on empty dataset: {dataset}')
            result = None
        results.append(result)
    return results


def _diversity_validation(
    dataset: AlignmentDataset,
    num_workers: int,
    ngram: int,
    n_references: Optional[int],
) -> Dict[str, float]:
    weight = [1.0 / ngram for _ in range(ngram)]
    prompts = [sample.prompt for sample in dataset.samples]
    chosens = [sample.chosen for sample in dataset.samples]
    rejected = [sample.rejected for sample in dataset.samples]

    results: Dict[str, List[float]] = {}
    logging.info('Computing prompt diversity')
    results['prompt_diversity'] = _compute_diversity(
        prompts, weight, num_workers, n_references
    )

    logging.info('Computing chosen response diversity')
    results['chosen_diversity'] = _compute_diversity(
        chosens, weight, num_workers, n_references
    )

    logging.info('Computing rejected response diversity')
    results['rejected_diversity'] = _compute_diversity(
        rejected, weight, num_workers, n_references
    )
    return _compute_statistics(results)


def _compute_diversity(
    response_set: List[str],
    weight: List[float],
    num_workers: int,
    n_references: Optional[int],
) -> List[float]:
    if 0 <= len(response_set) < 2:
        return len(response_set) * [0.0]

    logging.info('Tokenizing responses')
    tokenizer = _get_tokenizer()
    tokenized_responses = [tokenizer(sentence) for sentence in response_set]

    with mp.Pool(num_workers) as pool:
        return [
            score
            for score in tqdm.tqdm(
                pool.imap_unordered(
                    _diversity_score_wrapper,
                    [
                        [tokenized_responses, i, weight, n_references]
                        for i in range(len(tokenized_responses))
                    ],
                    chunksize=len(tokenized_responses) // num_workers,
                ),
                total=len(tokenized_responses),
            )
        ]


def _diversity_score_wrapper(args: List[Any]) -> float:
    return _diversity_score(*args)


def _diversity_score(
    responses: List[str], i: int, weight: List[str], n_references: Optional[int]
) -> float:
    if n_references is not None:
        sample_indices = np.random.choice(
            [idx for idx in range(len(responses)) if idx != i],
            size=n_references,
        )
        references = [responses[idx] for idx in sample_indices]
    else:
        references = responses

    score = sentence_bleu(
        references,
        responses[i],
        weight,
        smoothing_function=SmoothingFunction().method1,
    )
    return 1 - score


def _compute_statistics(results: Dict[str, List[float]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        statistics[f'{metric}_mean'] = float(np.mean(values))
        statistics[f'{metric}_median'] = float(np.median(values))
        statistics[f'{metric}_min'] = float(np.min(values))
        statistics[f'{metric}_max'] = float(np.max(values))
    return statistics


def _get_tokenizer() -> Callable[[str], List[str]]:
    return nltk.word_tokenize


def _download_nltk_resources() -> None:
    required_resources = ['punkt_tab']
    for resource in required_resources:
        logging.info(f'Downloading NLTK: {resource}')
        nltk.download(resource)
        logging.info(f'Downloaded NLTK: {resource}')
