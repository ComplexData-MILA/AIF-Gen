from abc import ABC, abstractmethod
from typing import Any

from aif_gen.typing import Dataset


class DatasetTransform(ABC):
    r"""Base class for transforming Alignment Datasets."""

    @abstractmethod
    def apply(self, dataset: Dataset, in_place: bool = False) -> Dataset:
        r"""Apply the transform onto a dataset.

        Args:
            dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
            in_place: Whether to apply the transform in-place or return a new dataset.

        Returns:
            Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.
        """

    def __call__(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        return self.apply(dataset, *args, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__
