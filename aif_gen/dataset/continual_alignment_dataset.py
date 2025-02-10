import json
from typing import Any, Dict, List, Union

from aif_gen.dataset.alignment_sample import AlignmentDatasetSample

from .alignment_dataset import AlignmentDataset


class ContinualAlignmentDataset:
    r"""Container object for a Continual Alignment Dataset.

    Args:
        datasets (List[ContinualAlignmentDataset]): Temporal list of AlignmentDatasets constituents.
    """

    def __init__(self, datasets: List[AlignmentDataset]) -> None:
        self._datasets = datasets

    @property
    def datasets(self) -> List[AlignmentDataset]:
        r"""List[AlignmentDataset]: The list of AlignmentDataset constituents."""
        return self._datasets

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
        self, key: Union[slice, int]
    ) -> Union[AlignmentDatasetSample, List[AlignmentDatasetSample]]:
        # Indexing based on **samples** across datasets (not into datasets themselves)
        all_samples = []  # This should probably be cached
        for dataset in self.datasets:
            all_samples.extend(dataset.samples)
        return all_samples[key]

    def append(self, dataset: AlignmentDataset) -> None:
        r"""Append a single AlignmentDataset to the ConitnualAlignmentDataset.

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
        r"""Append multiple AlignmentDataset's to the ConitnualAlignmentDataset.

        Args:
            datasets (List[AlignmentDataset]): The new datasets to add.

        Raises:
            TypeError: if any dataset is not of type AlignmentDataset.
        """
        for dataset in datasets:
            self.append(dataset)

    def to_json(self, file_path: str) -> None:
        r"""Save the ContinualAlignmentDataset to a json file.

        Note: Uses to_dict() under the hood to get a dictionary representation.

        Args:
            file_path (str): The os.pathlike object to write to.
        """
        dataset_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the ContinualAlignmentDataset to dictionary represenetation.

        Note: This method is the functional inverse of ContinualAlignmentDataset.from_dict().

        Returns:
            Dict[str, Any]: The dictionary representation of the ContinualAlignmentDataset.
        """
        dataset_dict: Dict[str, List[Any]] = {'datasets': []}
        for dataset in self.datasets:
            dataset_dict['datasets'].append(dataset.to_dict())
        return dataset_dict

    @classmethod
    def from_json(cls, file_path: str) -> 'ContinualAlignmentDataset':
        r"""Load the ContinualAlignmentDataset from a json file.

        Note: Uses ContinualAlignmentDataset.from_dict() under the hood to parse the representation.

        Args:
            file_path (str): The os.pathlike object to read from.

        Returns:
            ContinualAlignmentDataset: The newly constructed ContinualAlignmentDataset.
        """
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)
        return cls.from_dict(dataset_dict)

    @classmethod
    def from_dict(cls, dataset_dict: Dict[str, Any]) -> 'ContinualAlignmentDataset':
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


class DebugContinualDataset(ContinualAlignmentDataset):

    def __init__(self, ) -> None:
        from datasets import load_dataset
        datasets = [
            {"train": load_dataset("trl-lib/ultrafeedback_binarized", split="train").select(range(100)),
             'test': load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(100)),
             'descriptiveness': load_dataset("trl-lib/ultrafeedback-prompt", split="train").select(range(200))},
            {"train": load_dataset("trl-lib/ultrafeedback_binarized", split="train").select(range(100, 200)),
             'test': load_dataset("trl-lib/ultrafeedback_binarized", split="test").select(range(100, 200)),
             'descriptiveness': load_dataset("trl-lib/ultrafeedback-prompt", split="train").select(range(200, 400))},
            {"train": load_dataset("Anthropic/hh-rlhf", split="train").select(range(100)),
             'test': load_dataset("Anthropic/hh-rlhf", split="test").select(range(100)),
             'descriptiveness': load_dataset("trl-lib/ultrafeedback-prompt", split="train").select(range(400, 600))},
        ]
        super().__init__(datasets)


class ContinualUltrafeedback2AnthropicDataset(ContinualAlignmentDataset):

    def __init__(self, ) -> None:
        from datasets import load_dataset
        datasets =  [
            load_dataset("trl-lib/ultrafeedback_binarized"),
            load_dataset("Anthropic/hh-rlhf")
            ]
        super().__init__(datasets)