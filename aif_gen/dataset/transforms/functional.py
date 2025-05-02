from .base import Dataset
from .preference_swap_transform import PreferenceSwapTransform
from .split_transform import SplitTransform


def preference_swap_transform(
    dataset: Dataset, swap_probability: float, in_place: bool = False
) -> Dataset:
    r"""Swaps the 'chosen' and 'rejected' responses for each sample in the dataset.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
        in_place: Whether to apply the transform in-place or return a new dataset.
        swap_probability (float): The independent probability of swapping responses for each sample in the dataset.

    Returns:
        Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.

    Raises:
        ValueError: If the swap probability is not in the range [0, 1].
    """
    transform = PreferenceSwapTransform(swap_probability)
    return transform(dataset, in_place=in_place)


def split_transform(
    dataset: Dataset, test_ratio: float, in_place: bool = False
) -> Dataset:
    r"""Splits a Dataset training data into train and test datasets.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
        in_place: Whether to apply the transform in-place or return a new dataset.
        test_ratio (float): The test ratio to split the dataset with.

    Returns:
        Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.

    Raises:
        ValueError: If a dataset in the Continual Dataset has test data.
    """
    transform = SplitTransform(test_ratio)
    return transform(dataset, in_place=in_place)
