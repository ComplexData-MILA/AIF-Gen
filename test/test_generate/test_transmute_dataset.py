import asyncio
import json

import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.dataset.alignment_sample import AlignmentDatasetSample
from aif_gen.generate import transmute_continual_dataset
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def async_mock_return(result):
    fut = asyncio.Future()
    fut.set_result(result)
    return fut


@pytest.fixture
def mock_client(mocker):
    mock_chosen_rejected_response = mocker.MagicMock(name='response')
    mock_chosen_rejected_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'OVERRIDE',
            'rejected': 'Mock rejected response',
        }
    )

    # Crude way of ensuring that every time we call 'chate_completion',
    # the mock client switches between a valid prompt_response schema
    # and a valid chosen_rejected_response schema.
    mock_client = mocker.MagicMock(name='client')
    mock_client.chat.completions.create.side_effect = [
        async_mock_return(mock_chosen_rejected_response),
    ] * 100

    return mock_client


@pytest.fixture
def mock_client_uncaught_exception(mocker):
    mock_response = mocker.MagicMock(name='response')
    mock_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'Mock chosen response',
            'rejected': 'Mock rejected response',
        }
    )
    mock_client = mocker.MagicMock(name='client')
    # Some uncaught exception (e.g. openai.NotFound)
    mock_client.chat.completions.create.side_effect = Exception
    return mock_client


@pytest.fixture
def mock_semaphore(mocker):
    mock_sem = mocker.MagicMock(name='async_semaphore')
    mock_sem.return_value.__aenter__.return_value = None
    return mock_sem


@pytest.fixture
def mock_model():
    return 'GPT-1337'


@pytest.fixture
def mock_dataset():
    task_1 = AlignmentTask(
        Domain([DomainComponent('Foo', seed_words=['seed'])]), 'bar', 'baz'
    )
    task_2 = AlignmentTask(
        Domain([DomainComponent('Foo2', seed_words=['seed'])]), 'bar2', 'baz2'
    )

    samples_1 = [AlignmentDatasetSample('prompt', 'chosen', 'rejected')]
    samples_2 = [AlignmentDatasetSample('prompt 2', 'chosen 2', 'rejected 2')]

    return ContinualAlignmentDataset(
        [AlignmentDataset(task_1, samples_1), AlignmentDataset(task_2, samples_2)]
    )


@pytest.mark.asyncio
async def test_transmute_continual_dataset_uncaught_exception(
    mock_dataset, mock_model, mock_client_uncaught_exception, mock_semaphore
):
    with pytest.raises(Exception):
        continual_dataset = await transmute_continual_dataset(
            mock_dataset, mock_model, mock_client_uncaught_exception, mock_semaphore
        )
        assert continual_dataset is None


@pytest.mark.asyncio
async def test_transmute_continual_dataset(
    mock_dataset, mock_model, mock_client, mock_semaphore
):
    continual_dataset = await transmute_continual_dataset(
        mock_dataset, mock_model, mock_client, mock_semaphore
    )
    assert isinstance(continual_dataset, ContinualAlignmentDataset)
    assert len(continual_dataset.datasets) == len(mock_dataset.datasets)

    for i in range(len(continual_dataset.datasets)):
        dataset = continual_dataset.datasets[i]
        assert isinstance(dataset, AlignmentDataset)

        original = mock_dataset.datasets[i]
        assert len(dataset) == len(original)
        assert dataset.task.to_dict() == original.task.to_dict()
        for sample_idx, sample in enumerate(dataset.samples):
            assert isinstance(sample, AlignmentDatasetSample)
            expected = original[sample_idx]
            expected.rejected = 'OVERRIDE'
            assert sample == expected
