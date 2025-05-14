from typing import Dict, List

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.typing import Dataset

from ._stop_words import remove_stop_words as rsw


def count_validation(
    dataset: Dataset, remove_stop_words: bool = False
) -> List[Dict[str, int | float]]:
    r"""Count the number of 'unique' samples and average phrase length in the dataset.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to validate.
        remove_stop_words (bool): If true, applies stop word removal before computing dataset counts.

    Returns:
        List[Dict[str, int]]: For every AligmentDataset, returns a dictionary with the following entries:

        'samples'               -> int: The total number of samples in the AlignmentDataset.
        'unique_samples'        -> int: The number of unique samples in the AlignmentDataset.
        'unique_prompts'        -> int: The number of unique prompts in the AlignmentDataset.
        'unique_chosen'         -> int: The number of unique chosen responses in the AlignmentDataset.
        'unique_rejected'       -> int: The number of unique rejected responses in the AlignmentDataset.
        'avg_prompt_length'     -> float: The average length of prompts in the AlignmentDataset.
        'avg_chosen_length'     -> float: The average length of chosen responses in the AlignmentDataset.
        'avg_rejected_length'   -> float: The average length of rejected responses in the AlignmentDataset.

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
        results.append(_count_validation(dataset, remove_stop_words))
    return results


def _count_validation(
    dataset: AlignmentDataset, remove_stop_words: bool
) -> Dict[str, int | float]:
    samples, prompts, chosen, rejected = set(), set(), set(), set()
    for sample in dataset.samples:
        samples.add(rsw(str(sample)) if remove_stop_words else str(sample))
        prompts.add(rsw(sample.prompt) if remove_stop_words else sample.prompt)
        chosen.add(rsw(sample.chosen) if remove_stop_words else sample.chosen)
        rejected.add(rsw(sample.rejected) if remove_stop_words else sample.rejected)

    def _avg_word_count(strs: List[str]) -> float:
        return sum(len(s.split()) for s in strs) / len(strs) if len(strs) else 0

    return {
        'sample': dataset.num_samples,
        'unique_samples': len(samples),
        'unique_prompts': len(prompts),
        'unique_chosen': len(chosen),
        'unique_rejected': len(rejected),
        'avg_prompt_length': _avg_word_count([x.prompt for x in dataset.samples]),
        'avg_chosen_length': _avg_word_count([x.chosen for x in dataset.samples]),
        'avg_rejected_length': _avg_word_count([x.rejected for x in dataset.samples]),
    }
