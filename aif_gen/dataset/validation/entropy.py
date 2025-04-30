import typing
from collections import Counter
from typing import Dict, List

import numpy as np

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset

from ._stop_words import remove_stop_words as rsw


def entropy_validation(
    dataset: Dataset, remove_stop_words: bool = False
) -> List[Dict[str, float]]:
    r"""Report various entropy measures on tokens in the dataset samples.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        remove_stop_words (bool): If true, applies stop word removal before computing dataset counts.

    Returns:
        List[Dict[str, int]]: For every AligmentDataset, returns a dictionary with the following entries:

        'token_entropy'     -> float: The entropy across tokens (prompts and responses combined) for all samples in the AlignmentDataset.
        'prompt_entropy'    -> float: The entropy across prompts in samples of the AlignmentDataset.
        'chosen_entropy'    -> float: The entropy across chosen responses in samples of the AlignmentDataset.
        'rejected_entropy'  -> float: The entropy across rejected responses in the samples of the AlignmentDataset.

    Note:
        If the input dataset is an AlignmentDataset (non-continual), this function
        returns a 1 element list with the relevant statistics.
    """
    if isinstance(dataset, AlignmentDataset):
        datasets = [dataset]
    else:
        # This assert is here to make mypy happy
        assert isinstance(dataset, ContinualAlignmentDataset)
        datasets = dataset.datasets

    results = []
    for dataset in datasets:
        results.append(_entropy_validation(dataset, remove_stop_words))
    return results


def _entropy_validation(
    dataset: AlignmentDataset, remove_stop_words: bool
) -> Dict[str, float]:
    token_freq: typing.Counter[str] = Counter()
    prompts_freq: typing.Counter[str] = Counter()
    chosen_freq: typing.Counter[str] = Counter()
    rejected_freq: typing.Counter[str] = Counter()

    for sample in dataset.samples:
        sample_str = rsw(str(sample)) if remove_stop_words else str(sample)
        prompt_str = rsw(sample.prompt) if remove_stop_words else sample.prompt
        chosen_str = rsw(sample.chosen) if remove_stop_words else sample.chosen
        rejected_str = rsw(sample.rejected) if remove_stop_words else sample.rejected

        token_freq.update(sample_str.split())
        prompts_freq.update(prompt_str.split())
        chosen_freq.update(chosen_str.split())
        rejected_freq.update(rejected_str.split())

    return {
        'token_entropy': _compute_entropy(token_freq),
        'prompt_entropy': _compute_entropy(prompts_freq),
        'chosen_entropy': _compute_entropy(chosen_freq),
        'rejected_entropy': _compute_entropy(rejected_freq),
    }


def _compute_entropy(word_freq: Counter) -> float:
    # Normalize frequency counts to probabilities
    norm_counts_ = []
    total = sum(word_freq.values(), 0.0)
    for word in word_freq:
        norm_counts_.append(word_freq[word] / total)

    norm_counts = np.array(norm_counts_)
    return float(-(norm_counts * np.log(norm_counts)).sum())
