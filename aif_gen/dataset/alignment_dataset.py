from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from typing import Any, Dict, List

from datasets import Dataset
from pydantic import Field
from pydantic.dataclasses import dataclass

from aif_gen.task import AlignmentTask

from .alignment_sample import AlignmentDatasetSample


@dataclass(slots=True)
class AlignmentDataset:
    r"""Container object for an Alignment Dataset.

    Args:
        task (AligmnentTask): The AlignmentTask associated with the dataset.
        samples (List[AlignmentDatasetSample]): The samples in this AlignmentDataset.
        train_frac (float): Fraction of samples that belong to the training split.

    Raises:
        ValueError: If train_frac is not in the interval [0, 1.0]
    """

    task: AlignmentTask = Field(frozen=True)
    samples: List[AlignmentDatasetSample] = Field(frozen=True)
    train_frac: float = Field(default=1.0, ge=0, le=1)

    @property
    def test_frac(self) -> float:
        r"""Fraction of samples that belong to the testing split."""
        return 1.0 - self.train_frac

    @property
    def train(self) -> List[AlignmentDatasetSample]:
        r"""List[AlignmentDatasetSample]: The list of training samples associated with the AlignmentDataset."""
        return self.samples[: self.num_train_samples]

    @property
    def test(self) -> List[AlignmentDatasetSample]:
        r"""List[AlignmentDatasetSample]: The list of testing samples associated with the AlignmentDataset."""
        return self.samples[self.num_train_samples :]

    @property
    def num_samples(self) -> int:
        r"""int: The number of samples associated with the AlignmentDataset."""
        return len(self.samples)

    @property
    def num_train_samples(self) -> int:
        r"""int: The number of training samples associated with the AlignmentDataset."""
        return int(self.train_frac * len(self.samples))

    @property
    def num_test_samples(self) -> int:
        r"""int: The number of test samples associated with the AlignmentDataset."""
        return self.num_samples - self.num_train_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, key: slice | int
    ) -> AlignmentDatasetSample | List[AlignmentDatasetSample]:
        # Slicing directly on the samples
        return self.samples[key]

    def to_json(self, file_path: str | pathlib.Path) -> None:
        r"""Save the AlignmentDataset to a json file.

        Note: Uses to_dict() under the hood to get a dictionary representation.

        Args:
            file_path (Union[str, pathlib.Path]): The os.pathlike object to write to.
        """
        dataset_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(dataset_dict, f)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the AlignmentDataset to dictionary represenetation.

        Returns:
            Dict[str, Any]: The dictionary representation of the AlignmentDataset.
        """
        dataset_dict: Dict[str, Any] = {}
        dataset_dict['task'] = self.task.to_dict()
        dataset_dict['train'] = [asdict(sample) for sample in self.train]
        dataset_dict['test'] = [asdict(sample) for sample in self.test]
        return dataset_dict

    @classmethod
    def from_json(cls, file_path: str | pathlib.Path) -> AlignmentDataset:
        r"""Load the AlignmentDataset from a json file.

        Note: Uses AlignmentDataset.from_dict() under the hood to parse the representation.

        Args:
            file_path (Union[str, pathlib.Path]): The os.pathlike object to read from.

        Returns:
            AlignmentDataset: The newly constructed AlignmentDataset.
        """
        with open(file_path, 'r') as f:
            dataset_dict = json.load(f)
        return cls.from_dict(dataset_dict)

    @classmethod
    def from_dict(cls, dataset_dict: Dict[str, Any]) -> AlignmentDataset:
        r"""Construct an AlignmentDataset from dictionary representation.

        Note:
            Expects 'task', and 'train', 'test' keys to be present in the dictionary.
            The 'task' value should be parsable by AlignmentTask.from_dict().
            The 'train' and 'test' value should be a list of dictionaries, each of which
            are parsable by AlignmentDatasetSample.

        Args:
            dataset_dict (Dict[str, Any]): The dictionary that encodes the AlignmentDataset.

        Returns:
            AlignmentDataset: The newly constructed AlignmentDataset.

        Raises:
            ValueError: If the input dictionary is missing any required keys.
        """
        task = AlignmentTask.from_dict(dataset_dict['task'])
        samples = []
        for sample in dataset_dict['train']:
            samples.append(AlignmentDatasetSample(**sample))
        num_train_samples = len(samples)

        for sample in dataset_dict['test']:
            samples.append(AlignmentDatasetSample(**sample))

        train_frac = num_train_samples / len(samples)
        return cls(task, samples, train_frac)

    def to_hf_compatible(self) -> Dict[str, Dataset]:
        r"""Convert the AlignmentDataset to a dictionary compatible with HuggingFace datasets.

        Returns:
            Dict[str, Dataset]: The dictionary compatible with HuggingFace datasets.
        """
        hf_dict: Dict[str, Dataset] = {
            'train': Dataset.from_dict(
                {
                    'prompt': [sample.prompt for sample in self.train],
                    'chosen': [sample.chosen for sample in self.train],
                    'rejected': [sample.rejected for sample in self.train],
                },
                split='train',
            ),
            'test': Dataset.from_dict(
                {
                    'prompt': [sample.prompt for sample in self.test],
                    'chosen': [sample.chosen for sample in self.test],
                    'rejected': [sample.rejected for sample in self.test],
                },
                split='test',
            ),
        }
        return hf_dict
