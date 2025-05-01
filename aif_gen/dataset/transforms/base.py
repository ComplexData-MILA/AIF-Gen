from abc import ABC, abstractmethod
from typing import Any

from aif_gen.dataset import ContinualAlignmentDataset
from aif_gen.typing import Dataset


class DatasetTransform(ABC):
    r"""Base class for transforming Alignment Datasets."""

    @abstractmethod
    def apply(
        self, dataset: Dataset, in_place: bool = False, *args: Any, **kwargs: Any
    ) -> Dataset:
        r"""Apply the transform onto a dataset.

        Args:
            dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
            in_place: Whether to apply the transform in-place or return a new dataset.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.

        Returns:
            Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.
        """

    def __call__(self, dataset: Dataset, *args: Any, **kwargs: Any) -> Dataset:
        return self.apply(dataset, *args, **kwargs)

    def __str__(self) -> str:
        r"""Returns the type of Dataset transform."""
        return self.__class__.__name__

    def _is_dataset_continual(self, dataset: Dataset) -> bool:
        return isinstance(dataset, ContinualAlignmentDataset)
