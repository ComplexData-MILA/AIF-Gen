from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from datasets import Dataset
from pydantic.dataclasses import dataclass

from .alignment_dataset import AlignmentDataset
from .alignment_sample import AlignmentDatasetSample


@dataclass(slots=True)
class ContinualAlignmentDataset:
    r"""Container object for a Continual Alignment Dataset.

    Args:
        datasets (List[ContinualAlignmentDataset]): Temporal list of AlignmentDatasets constituents.
    """

    datasets: List[AlignmentDataset]

    @property
    def num_datasets(self) -> int:
        r"""int: The number of AlignmentDataset constituents."""
        return len(self.datasets)

    @property
    def num_samples(self) -> int:
        r"""int: The total number of samples acros all AlignmentDataset constituents."""
        return sum(len(dataset) for dataset in self.datasets)

    def __len__(self) -> int:
        r"""int: The total number of samples acros all AlignmentDataset constituents."""
        return self.num_samples

    def __getitem__(
        self, key: slice | int
    ) -> AlignmentDatasetSample | List[AlignmentDatasetSample]:
        # Indexing based on **samples** across datasets (not into datasets themselves)
        all_samples = []  # This should probably be cached
        for dataset in self.datasets:
            all_samples.extend(dataset.samples)
        return all_samples[key]

    def append(self, dataset: AlignmentDataset) -> None:
        r"""Append a single AlignmentDataset to the ContinualAlignmentDataset.

        Args:
            dataset (AlignmentDataset): The new dataset to add.

        Raises:
            TypeError: if the sample is not of type AlignmentDataset.
        """
        if isinstance(dataset, AlignmentDataset):
            self.datasets.append(dataset)
        else:
            raise TypeError(
                f'Dataset: {dataset} must be of type AlignmentDataset but got {dataset.__class__.__name__}'
            )

    def extend(self, datasets: List[AlignmentDataset]) -> None:
        r"""Append multiple AlignmentDataset's to the ContinualAlignmentDataset.

        Args:
            datasets (List[AlignmentDataset]): The new datasets to add.

        Raises:
            TypeError: if any dataset is not of type AlignmentDataset.
        """
        for dataset in datasets:
            self.append(dataset)

    def to_json(self, file_path: str | pathlib.Path) -> None:
        r"""Save the ContinualAlignmentDataset to a json file.

        Note: Uses to_dict() under the hood to get a dictionary representation.

        Args:
            file_path (Union[str, pathlib.Path]): The os.pathlike object to write to.
        """
        dataset_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the ContinualAlignmentDataset to dictionary represenetation.

        Returns:
            Dict[str, Any]: The dictionary representation of the ContinualAlignmentDataset.
        """
        dataset_dict: Dict[str, List[Any]] = {'datasets': []}
        for dataset in self.datasets:
            dataset_dict['datasets'].append(dataset.to_dict())
        return dataset_dict

    @classmethod
    def from_json(cls, file_path: str | pathlib.Path) -> ContinualAlignmentDataset:
        r"""Load the ContinualAlignmentDataset from a json file.

        Note: Uses ContinualAlignmentDataset.from_dict() under the hood to parse the representation.

        Args:
            file_path (Union[str, pathlib.Path]): The os.pathlike object to read from.

        Returns:
            ContinualAlignmentDataset: The newly constructed ContinualAlignmentDataset.
        """
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)
        return cls.from_dict(dataset_dict)

    @classmethod
    def from_dict(cls, dataset_dict: Dict[str, Any]) -> ContinualAlignmentDataset:
        r"""Construct a ContinualAlignmentDataset from dictionary representation.

        Note:
            Expects 'datasets' key to be present in the dictionary. The value is a list
            of dictionaries, each parsable by AlignmentDataset.from_dict().

        Args:
            dataset_dict (Dict[str, Any]): The dictionary that encodes the ContinualAlignmentDataset.

        Returns:
            ContinualAlignmentDataset: The newly constructed ContinualAlignmentDataset.

        Raises:
            ValueError: If the input dictionary is missing any required keys.
        """
        datasets = []
        for dataset in dataset_dict['datasets']:
            datasets.append(AlignmentDataset.from_dict(dataset))
        return cls(datasets)

    def to_hf_compatible(self) -> List[Dict[str, Dataset]]:
        r"""Convert the ContinualAlignmentDataset to a list of dictionaries compatible with HuggingFace datasets.

        Returns:
            List[Dict[str, Dataset]]: The list of dictionaries compatible with HuggingFace datasets.
        """
        return [dataset.to_hf_compatible() for dataset in self.datasets]
