from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.task.alignment_task import AlignmentTask
from aif_gen.task.domain import Domain
from aif_gen.validation import count_validation


def test_count_validation_all_unique():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 1', 'Winning Response A 1', 'Losing Response A 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt B 1', 'Winning Response B 1', 'Losing Response B 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt C 1', 'Winning Response C 1', 'Losing Response C 1'
        ),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    expected_counts = [
        {
            'sample': 3,
            'unique_samples': 3,
            'unique_prompts': 3,
            'unique_chosen': 3,
            'unique_rejected': 3,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        }
    ]
    assert count_validation(dataset) == expected_counts


def test_count_validation_all_same_prompts():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 2', 'Winning Response A 2', 'Losing Response A 2'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 2', 'Winning Response B 2', 'Losing Response B 2'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 2', 'Winning Response C 2', 'Losing Response C 2'
        ),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    expected_counts = [
        {
            'sample': 3,
            'unique_samples': 3,
            'unique_prompts': 1,
            'unique_chosen': 3,
            'unique_rejected': 3,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        }
    ]
    assert count_validation(dataset) == expected_counts


def test_count_validation_all_same_responses():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 3', 'Winning Response A 3', 'Losing Response B 3'
        ),
        AlignmentDatasetSample(
            'Mock prompt B 3', 'Winning Response A 3', 'Losing Response B 3'
        ),
        AlignmentDatasetSample(
            'Mock prompt C 3', 'Winning Response A 3', 'Losing Response B 3'
        ),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    expected_counts = [
        {
            'sample': 3,
            'unique_samples': 3,
            'unique_prompts': 3,
            'unique_chosen': 1,
            'unique_rejected': 1,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        }
    ]
    assert count_validation(dataset) == expected_counts


def test_count_validation_all_same_everything():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 4', 'Winning Response A 4', 'Losing Response A 4'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 4', 'Winning Response A 4', 'Losing Response A 4'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 4', 'Winning Response A 4', 'Losing Response A 4'
        ),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    expected_counts = [
        {
            'sample': 3,
            'unique_samples': 1,
            'unique_prompts': 1,
            'unique_chosen': 1,
            'unique_rejected': 1,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        }
    ]
    assert count_validation(dataset) == expected_counts


def test_count_countinual_dataset():
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 1', 'Winning Response A 1', 'Losing Response A 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt B 1', 'Winning Response B 1', 'Losing Response B 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt C 1', 'Winning Response C 1', 'Losing Response C 1'
        ),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset_one = AlignmentDataset(task=mock_task, samples=samples)

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 2', 'Winning Response A 2', 'Losing Response A 2'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 2', 'Winning Response B 2', 'Losing Response B 2'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 2', 'Winning Response C 2', 'Losing Response C 2'
        ),
    ]
    dataset_two = AlignmentDataset(task=mock_task, samples=samples)

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 3', 'Winning Response A 3', 'Losing Response B 3'
        ),
        AlignmentDatasetSample(
            'Mock prompt B 3', 'Winning Response A 3', 'Losing Response B 3'
        ),
        AlignmentDatasetSample(
            'Mock prompt C 3', 'Winning Response A 3', 'Losing Response B 3'
        ),
    ]
    dataset_three = AlignmentDataset(task=mock_task, samples=samples)

    samples = [
        AlignmentDatasetSample(
            'Mock prompt A 4', 'Winning Response A 4', 'Losing Response A 4'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 4', 'Winning Response A 4', 'Losing Response A 4'
        ),
        AlignmentDatasetSample(
            'Mock prompt A 4', 'Winning Response A 4', 'Losing Response A 4'
        ),
    ]
    dataset_four = AlignmentDataset(task=mock_task, samples=samples)

    dataset = ContinualAlignmentDataset(
        [dataset_one, dataset_two, dataset_three, dataset_four]
    )

    expected_counts = [
        {
            'sample': 3,
            'unique_samples': 3,
            'unique_prompts': 3,
            'unique_chosen': 3,
            'unique_rejected': 3,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        },
        {
            'sample': 3,
            'unique_samples': 3,
            'unique_prompts': 1,
            'unique_chosen': 3,
            'unique_rejected': 3,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        },
        {
            'sample': 3,
            'unique_samples': 3,
            'unique_prompts': 3,
            'unique_chosen': 1,
            'unique_rejected': 1,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        },
        {
            'sample': 3,
            'unique_samples': 1,
            'unique_prompts': 1,
            'unique_chosen': 1,
            'unique_rejected': 1,
            'avg_prompt_length': 4.0,
            'avg_chosen_length': 4.0,
            'avg_rejected_length': 4.0,
        },
    ]
    assert count_validation(dataset) == expected_counts


def test_count_validation_stop_words_removed():
    samples = [
        AlignmentDatasetSample(
            'i Mock prompt A 4', 'me Winning Response A 4', 'you Losing Response A 4'
        ),
        AlignmentDatasetSample(
            'be Mock prompt A 4', 'a Winning Response A 4', 'or Losing Response A 4'
        ),
        AlignmentDatasetSample(
            'with Mock prompt A 4', 'by Winning Response A 4', 'is Losing Response A 4'
        ),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    expected_counts = [
        {
            'sample': 3,
            'unique_samples': 1,
            'unique_prompts': 1,
            'unique_chosen': 1,
            'unique_rejected': 1,
            'avg_prompt_length': 5.0,
            'avg_chosen_length': 5.0,
            'avg_rejected_length': 5.0,
        }
    ]
    assert count_validation(dataset, remove_stop_words=True) == expected_counts
