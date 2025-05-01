import asyncio

import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.dataset.alignment_sample import AlignmentDatasetSample
from aif_gen.dataset.validation import llm_embedding_diversity
from aif_gen.task import AlignmentTask, Domain, DomainComponent

_BATCH_SIZE = 1
_EXP_KEYS = [
    'prompt_max',
    'prompt_mean',
    'prompt_median',
    'prompt_min',
    'chosen_max',
    'chosen_mean',
    'chosen_median',
    'chosen_min',
    'rejected_max',
    'rejected_mean',
    'rejected_median',
    'rejected_min',
]


def async_mock_return(result):
    fut = asyncio.Future()
    fut.set_result(result)
    return fut


@pytest.fixture
def mock_client(mocker):
    embedding_dim = 1024
    mock_response = mocker.MagicMock(name='data')
    embeddings = [mocker.MagicMock(name='data') for _ in range(_BATCH_SIZE)]
    for embedding in embeddings:
        embedding.embedding = [0.0] * embedding_dim
    mock_response.data = embeddings

    mock_client = mocker.MagicMock(name='client')
    mock_client.embeddings.create.side_effect = [
        async_mock_return(mock_response),
    ] * 100
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
async def test_llm_embedding_diversity_continual_dataset(
    mock_dataset, mock_model, mock_client, mock_semaphore
):
    results = await llm_embedding_diversity(
        mock_dataset, mock_model, mock_client, _BATCH_SIZE, mock_semaphore
    )
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert sorted(list(result.keys())) == sorted(_EXP_KEYS)
