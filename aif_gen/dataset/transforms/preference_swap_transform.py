import copy
from dataclasses import replace

import numpy as np

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset

from .base import Dataset, DatasetTransform


class PreferenceSwapTransform(DatasetTransform):
    r"""PreferenceSwapTransform swaps the 'chosen' and 'rejected' responses for each sample in the dataset.

    Args:
        dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
        swap_probability (float): The independent probability of swapping responses for each sample in the dataset.

    Returns:
        Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.

    Raises:
        ValueError: If the swap probability is not in the range [0, 1].
    """

    def __init__(self, swap_probability: float) -> None:
        self._validate_swap_probability(swap_probability)
        self._swap_probability = swap_probability

    @property
    def swap_probability(self) -> float:
        r"""float: The swap probability associated with the transform."""
        return self._swap_probability

    @swap_probability.setter
    def swap_probability(self, swap_probability: float) -> None:
        self._validate_swap_probability(swap_probability)
        self._swap_probability = swap_probability

    def apply(self, dataset: Dataset, in_place: bool = False) -> Dataset:
        r"""Swap the 'chosen' and 'rejected' responses for each sample in the dataset.

        Args:
            dataset (Union[ContinualAlignmentDataset, AlignmentDataset]): The dataset to transform.
            in_place: Whether to apply the transform in-place or return a new dataset.

        Returns:
            Union[ContinualAlignmentDataset, AlignmentDataset]: The transformed dataset.
        """
        if self.swap_probability == 0:
            return dataset if in_place else copy.deepcopy(dataset)

        if self._is_dataset_continual(dataset):
            # This assert is here to make mypy happy
            assert isinstance(dataset, ContinualAlignmentDataset)
            if in_place:
                for i in range(dataset.num_datasets):
                    dataset.datasets[i] = self._apply(dataset.datasets[i], in_place)
                return dataset
            else:
                return ContinualAlignmentDataset(
                    [self._apply(data, in_place) for data in dataset.datasets]
                )
        else:
            # This assert is here to make mypy happy
            assert isinstance(dataset, AlignmentDataset)
            return self._apply(dataset, in_place)

    def _apply(self, dataset: AlignmentDataset, in_place: bool) -> AlignmentDataset:
        swap_outcomes = np.random.binomial(
            n=1, p=self.swap_probability, size=len(dataset)
        )
        if in_place:
            return self._apply_inplace(dataset, swap_outcomes)
        else:
            return self._apply_copy(dataset, swap_outcomes)

    def _apply_inplace(
        self, dataset: AlignmentDataset, swap_outcomes: np.ndarray
    ) -> AlignmentDataset:
        for i in range(len(dataset)):
            if swap_outcomes[i]:
                chosen = dataset.samples[i].chosen
                rejected = dataset.samples[i].rejected
                dataset.samples[i].chosen = rejected
                dataset.samples[i].rejected = chosen
        return dataset

    def _apply_copy(
        self, dataset: AlignmentDataset, swap_outcomes: np.ndarray
    ) -> AlignmentDataset:
        transformed_samples = []
        for i, sample in enumerate(dataset.samples):
            if swap_outcomes[i]:
                transformed_sample = replace(
                    sample, chosen=sample.rejected, rejected=sample.chosen
                )
            else:
                transformed_sample = replace(sample)  # Simply copy over
            transformed_samples.append(transformed_sample)
        return AlignmentDataset(
            task=dataset.task,
            samples=transformed_samples,
            train_frac=dataset.train_frac,
        )

    def _validate_swap_probability(self, swap_probability: float) -> None:
        if not 0 <= swap_probability <= 1:
            raise ValueError(
                f'Expected a swap probability in the range [0, 1] but got: {swap_probability}'
            )
