from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.task.alignment_task import AlignmentTask
from aif_gen.task.domain import Domain
from aif_gen.validation import entropy_validation


def test_entropy_validation_all_unique():
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
    expected_entropy = [
        {
            'chosen_entropy': 1.660947433286918,
            'prompt_entropy': 1.660947433286918,
            'rejected_entropy': 1.660947433286918,
        }
    ]
    assert entropy_validation(dataset) == expected_entropy


def test_entropy_validation_all_same_prompts():
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
    expected_entropy = [
        {
            'chosen_entropy': 1.660947433286918,
            'prompt_entropy': 1.3862943611198906,
            'rejected_entropy': 1.660947433286918,
        }
    ]
    assert entropy_validation(dataset) == expected_entropy


def test_entropy_validation_all_same_responses():
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
    expected_entropy = [
        {
            'chosen_entropy': 1.3862943611198906,
            'prompt_entropy': 1.660947433286918,
            'rejected_entropy': 1.3862943611198906,
        }
    ]
    assert entropy_validation(dataset) == expected_entropy


def test_entropy_validation_all_same_everything():
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
    expected_entropy = [
        {
            'chosen_entropy': 1.3862943611198906,
            'prompt_entropy': 1.3862943611198906,
            'rejected_entropy': 1.3862943611198906,
        }
    ]
    assert entropy_validation(dataset) == expected_entropy


def test_entropy_validation_no_response_entropy():
    samples = [
        AlignmentDatasetSample('Mock prompt A 5', 'foo foo foo', 'bar bar bar'),
        AlignmentDatasetSample('Mock prompt B 5', 'foo foo foo', 'bar bar bar'),
        AlignmentDatasetSample('Mock prompt C 5', 'foo foo foo', 'bar bar bar'),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    expected_entropy = [
        {
            'chosen_entropy': -0.0,
            'prompt_entropy': 1.660947433286918,
            'rejected_entropy': -0.0,
        }
    ]
    assert entropy_validation(dataset) == expected_entropy


def test_entropy_validation_no_prompt_entropy():
    samples = [
        AlignmentDatasetSample(
            'foo foo foo', 'Winning Response A 6', 'Losing Response C 6'
        ),
        AlignmentDatasetSample(
            'foo foo foo', 'Winning Response B 6', 'Losing Response C 6'
        ),
        AlignmentDatasetSample(
            'foo foo foo', 'Winning Response C 6', 'Losing Response C 6'
        ),
    ]
    mock_task = AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    expected_entropy = [
        {
            'chosen_entropy': 1.660947433286918,
            'prompt_entropy': -0.0,
            'rejected_entropy': 1.3862943611198906,
        }
    ]
    assert entropy_validation(dataset) == expected_entropy


def test_entropy_countinual_dataset():
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

    samples = [
        AlignmentDatasetSample('Mock prompt A 5', 'foo foo foo', 'bar bar bar'),
        AlignmentDatasetSample('Mock prompt B 5', 'foo foo foo', 'bar bar bar'),
        AlignmentDatasetSample('Mock prompt C 5', 'foo foo foo', 'bar bar bar'),
    ]
    dataset_five = AlignmentDataset(task=mock_task, samples=samples)

    samples = [
        AlignmentDatasetSample(
            'foo foo foo', 'Winning Response A 6', 'Losing Response C 6'
        ),
        AlignmentDatasetSample(
            'foo foo foo', 'Winning Response B 6', 'Losing Response C 6'
        ),
        AlignmentDatasetSample(
            'foo foo foo', 'Winning Response C 6', 'Losing Response C 6'
        ),
    ]
    dataset_six = AlignmentDataset(task=mock_task, samples=samples)
    expected_entropy = []

    dataset = ContinualAlignmentDataset(
        [
            dataset_one,
            dataset_two,
            dataset_three,
            dataset_four,
            dataset_five,
            dataset_six,
        ]
    )

    expected_entropy = [
        {
            'chosen_entropy': 1.660947433286918,
            'prompt_entropy': 1.660947433286918,
            'rejected_entropy': 1.660947433286918,
        },
        {
            'chosen_entropy': 1.660947433286918,
            'prompt_entropy': 1.3862943611198906,
            'rejected_entropy': 1.660947433286918,
        },
        {
            'chosen_entropy': 1.3862943611198906,
            'prompt_entropy': 1.660947433286918,
            'rejected_entropy': 1.3862943611198906,
        },
        {
            'chosen_entropy': 1.3862943611198906,
            'prompt_entropy': 1.3862943611198906,
            'rejected_entropy': 1.3862943611198906,
        },
        {
            'chosen_entropy': -0.0,
            'prompt_entropy': 1.660947433286918,
            'rejected_entropy': -0.0,
        },
        {
            'chosen_entropy': 1.660947433286918,
            'prompt_entropy': -0.0,
            'rejected_entropy': 1.3862943611198906,
        },
    ]
    assert entropy_validation(dataset) == expected_entropy


def test_entropy_validation_stop_words_removed():
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
    expected_entropy = [
        {
            'chosen_entropy': 1.0986122886681096,
            'prompt_entropy': 1.0986122886681096,
            'rejected_entropy': 1.0986122886681096,
        }
    ]
    assert entropy_validation(dataset, remove_stop_words=True) == expected_entropy
