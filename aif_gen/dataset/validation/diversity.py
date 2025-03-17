import logging
from typing import Dict, List, Optional
from math import ceil

import nltk
import numpy as np
import tqdm
from transformers import AutoTokenizer


from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset
from aif_gen.dataset.validation.jax_bleu_utils import (
    calculate_bleu_similarity,
    preprocess_for_jax_bleu,
)
from nltk.tokenize import word_tokenize


def diversity_validation(
    dataset: Dataset, batch_size: int, ngram: int = 3, n_references: Optional[int] = 20
) -> List[Optional[Dict[str, float]]]:
    r"""Report the inverse Self-BLEU score as a measure of diversity within the generated samples.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        batch_size (int): Number of examples to send to the jax-jit BLEU kernel at one time.
            A larger batch_size increases throughput, but also uses more memory.
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

    if isinstance(dataset, AlignmentDataset):
        datasets = [dataset]
    else:
        # This assert is here to make mypy happy
        assert isinstance(dataset, ContinualAlignmentDataset)
        datasets = dataset.datasets

    results: List[Optional[Dict[str, float]]] = []
    for dataset in datasets:
        if len(dataset):
            result = _diversity_validation(dataset, ngram, batch_size, n_references)
        else:
            logging.warning(f'Skipping diversity on empty dataset: {dataset}')
            result = None
        results.append(result)
    return results


def _diversity_validation(
    dataset: AlignmentDataset,
    ngram: int,
    batch_size: int,
    n_references: Optional[int],
) -> Dict[str, float]:
    weights = [1.0 / ngram for _ in range(ngram)]
    prompts = [sample.prompt for sample in dataset.samples]
    chosens = [sample.chosen for sample in dataset.samples]
    rejected = [sample.rejected for sample in dataset.samples]

    results: Dict[str, List[float]] = {}
    logging.info('Computing prompt diversity')
    results['prompt_diversity'] = _compute_diversity(
        prompts, weights, n_references, batch_size
    )

    logging.info('Computing chosen response diversity')
    results['chosen_diversity'] = _compute_diversity(
        chosens, weights, n_references, batch_size
    )

    logging.info('Computing rejected response diversity')
    results['rejected_diversity'] = _compute_diversity(
        rejected, weights, n_references, batch_size
    )
    return _compute_statistics(results)


def _compute_diversity(
    response_set: List[str],
    weights: List[float],
    n_references: Optional[int],
    batch_size: int,
) -> List[float]:
    if 0 <= len(response_set) < 2:
        return len(response_set) * [0.0]

    output: list[float] = []
    tokens = preprocess_for_jax_bleu(
        response_set,
        tokenizer=word_tokenize,
        max_length=512,
        min_length=len(weights),
    )
    num_valid_seqs = tokens.shape[0]
    num_batches = ceil(num_valid_seqs / batch_size)

    with tqdm.tqdm(
        total=num_valid_seqs, ncols=75, desc=f'batch size: {batch_size}'
    ) as _pbar:
        for _batch_idx in range(num_batches):
            # Take a different sample of references for each response batch.
            if n_references is not None:
                sample_indices = np.random.choice(
                    list(range(tokens.shape[0])),
                    size=n_references,
                )
                _reference_tokens = tokens[sample_indices, :]
            else:
                _reference_tokens = response_set

            _response_batch = tokens[
                _batch_idx * batch_size : (_batch_idx + 1) * batch_size,
                :,
            ]

            _pbar.set_description(f'{_response_batch.shape}, {_reference_tokens.shape}')

            output.extend(
                calculate_bleu_similarity(
                    _response_batch,
                    _reference_tokens,
                    pad_token_id=-1,
                    # n_gram_weights=weights,
                )
                .flatten()
                .tolist(),
            )
            _pbar.update(_response_batch.shape[0])

    return output


def _tokenize(
    sentences: list[str], min_num_tokens: int, tokenizer_name='bert-base-uncased'
) -> np.ndarray:
    """Tokenize list of sentences.

    Params:
        sentences: list of N str, one for each sentence
        min_num_tokens: int, minimum number of tokens in sentence for a sentence
            to be included.

    Returns:
        np.ndarray: (n, l) where l is the max token supported for the specified tokenizer.
            n is the number of sentences where num_tokens is at least min_num_tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    output = tokenizer(
        sentences,
        padding='max_length',
        return_tensors='np',
        add_special_tokens=False,
    ).input_ids

    # Always use -1 as the pad token
    output = np.where(output != tokenizer.pad_token_type_id, output, -1)
    rows_included = np.sum(output != -1, axis=-1) >= min_num_tokens

    return output[rows_included]


def _compute_statistics(results: Dict[str, List[float]]) -> Dict[str, float]:
    statistics: Dict[str, float] = {}
    for metric, values in results.items():
        statistics[f'{metric}_mean'] = float(np.mean(values))
        statistics[f'{metric}_median'] = float(np.median(values))
        statistics[f'{metric}_min'] = float(np.min(values))
        statistics[f'{metric}_max'] = float(np.max(values))
    return statistics


def _download_nltk_resources() -> None:
    required_resources = ['punkt_tab']
    for resource in required_resources:
        logging.info(f'Downloading NLTK: {resource}')
        nltk.download(resource)
        logging.info(f'Downloaded NLTK: {resource}')
