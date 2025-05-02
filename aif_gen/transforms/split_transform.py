from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset

from .base import Dataset, DatasetTransform


class SplitTransform(DatasetTransform):
    r"""SplitTransform splits the training data into train/test datasets.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
        test_ratio (float): The test ratio to split the dataset with.

    Returns:
        Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.

    Raises:
        ValueError: If the test ratio is not in the range [0, 1].
    """

    def __init__(self, test_ratio: float) -> None:
        self._validate_test_ratio(test_ratio)
        self._test_ratio = test_ratio

    @property
    def test_ratio(self) -> float:
        r"""float: The test ratio to split the dataset with."""
        return self._test_ratio

    @test_ratio.setter
    def test_ratio(self, test_ratio: float) -> None:
        self._validate_test_ratio(test_ratio)
        self._test_ratio = test_ratio

    def apply(self, dataset: Dataset, in_place: bool = False) -> Dataset:
        r"""Splits a ContinualAlignmentDataset's training data into train and test datasets.

        Args:
            dataset (ContinualAlignmentDataset): The dataset to split.
            in_place (bool): Whether to apply the transform in-place or return a new dataset.

        Returns:
            ContinualAlignmentDataset: The dataset with test data included.

        Raises:
            ValueError: If a dataset in the Continual Dataset has test data.
        """
        self._check_test_frac_empty(dataset)
        if isinstance(dataset, ContinualAlignmentDataset):
            if in_place:
                for i in range(dataset.num_datasets):
                    dataset.datasets[i].train_frac = 1 - self.test_ratio
                return dataset
            else:
                transformed_datasets = []
                for data in dataset.datasets:
                    transformed_datasets.append(
                        AlignmentDataset(
                            data.task,
                            data.samples,
                            train_frac=1 - self.test_ratio,
                        )
                    )
                return ContinualAlignmentDataset(transformed_datasets)
        else:
            # This assert is here to make mypy happy
            assert isinstance(dataset, AlignmentDataset)
            if in_place:
                dataset.train_frac = 1 - self.test_ratio
                return dataset
            else:
                return AlignmentDataset(
                    dataset.task, dataset.samples, train_frac=1 - self.test_ratio
                )

    def _validate_test_ratio(self, test_ratio: float) -> None:
        if not 0 <= test_ratio <= 1:
            raise ValueError(f'Test ratio must be in [0, 1], got: {test_ratio}')

    def _check_test_frac_empty(self, dataset: Dataset) -> None:
        if isinstance(dataset, ContinualAlignmentDataset):
            datasets = dataset.datasets
        else:
            assert isinstance(dataset, AlignmentDataset)
            datasets = [dataset]
        for dataset in datasets:
            if dataset.test_frac != 0:
                raise ValueError('AlignmentDataset cannot have test data for splitting')
