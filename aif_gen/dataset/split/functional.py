from aif_gen.dataset import ContinualAlignmentDataset


def split(
    dataset: ContinualAlignmentDataset, test_ratio: float
) -> ContinualAlignmentDataset:
    r"""Splits a ContinualAlignmentDataset's training data into train and test datasets.

    Args:
        dataset (ContinualAlignmentDataset): The dataset to split.
        test_ratio (float): The ratio of the first dataset to the second.

    Returns:
        ContinualAlignmentDataset: The dataset with test data included.

    Raises:
        ValueError: If a dataset in the Continual Dataset has test data.
    """
    for i in range(len(dataset.datasets)):
        if dataset.datasets[i].test_frac != 0:
            raise ValueError(
                f'The dataset at index {i} has test data. Please remove it before splitting.'
            )
    # just change the fractions
    for i in range(len(dataset.datasets)):
        dataset.datasets[i]._train_frac = 1 - test_ratio

    return dataset
