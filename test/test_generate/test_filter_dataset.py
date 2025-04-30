import asyncio
import json

import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.dataset.alignment_sample import AlignmentDatasetSample
from aif_gen.generate import filter_continual_alignment_dataset_style_normalize
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def async_mock_return(result):
    fut = asyncio.Future()
    fut.set_result(result)
    return fut


@pytest.fixture
def mock_client(mocker):
    mock_style_response = mocker.MagicMock(name='response')
    mock_style_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'Normalized chosen response',
            'rejected': 'Normalized rejected response with lower quality',
        }
    )

    # Configure mock client to return our normalized responses
    mock_client = mocker.MagicMock(name='client')
    mock_client.chat.completions.create.side_effect = [
        async_mock_return(mock_style_response),
    ] * 100

    return mock_client


@pytest.fixture
def mock_client_uncaught_exception(mocker):
    mock_client = mocker.MagicMock(name='client')
    # Some uncaught exception (e.g. openai.NotFound)
    mock_client.chat.completions.create.side_effect = Exception
    return mock_client


@pytest.fixture
def mock_semaphore(mocker):
    mock_sem = mocker.MagicMock(name='async_semaphore')
    mock_sem.__aenter__ = mocker.AsyncMock(return_value=None)
    mock_sem.__aexit__ = mocker.AsyncMock(return_value=None)
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

    # Create samples with different formatting styles to be normalized
    samples_1 = [
        AlignmentDatasetSample('prompt', 'Chosen response!!!', 'rejected response...')
    ]
    samples_2 = [AlignmentDatasetSample('prompt 2', 'CHOSEN 2', 'rejected 2')]

    return ContinualAlignmentDataset(
        [AlignmentDataset(task_1, samples_1), AlignmentDataset(task_2, samples_2)]
    )


@pytest.mark.asyncio
async def test_filter_continual_alignment_dataset_uncaught_exception(
    mock_dataset, mock_model, mock_client_uncaught_exception, mock_semaphore
):
    with pytest.raises(Exception):
        continual_dataset = await filter_continual_alignment_dataset_style_normalize(
            mock_dataset, mock_model, mock_client_uncaught_exception, mock_semaphore
        )
        assert continual_dataset is None


@pytest.mark.asyncio
async def test_filter_continual_alignment_dataset(
    mock_dataset, mock_model, mock_client, mock_semaphore
):
    continual_dataset = await filter_continual_alignment_dataset_style_normalize(
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
            # Prompt should be unchanged
            assert sample.prompt == original[sample_idx].prompt
            # Both chosen and rejected should be normalized
            assert sample.chosen == 'Normalized chosen response'
            assert sample.rejected == 'Normalized rejected response with lower quality'


@pytest.mark.asyncio
async def test_filter_continual_alignment_dataset_dry_run(
    mock_dataset, mock_model, mock_client, mock_semaphore
):
    continual_dataset = await filter_continual_alignment_dataset_style_normalize(
        mock_dataset, mock_model, mock_client, mock_semaphore, dry_run=True
    )
    # Dry run should return None
    assert continual_dataset is None
    # Should have called create once for dry run
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_filter_continual_alignment_dataset_with_caching(
    mock_dataset, mock_model, mock_client, mock_semaphore, mocker
):
    # Use AsyncMock for the async cache methods.
    mock_cache = mocker.MagicMock()
    mock_cache.get = mocker.AsyncMock(return_value=None)
    mock_cache.set = mocker.AsyncMock(return_value=None)
    mock_cache.close = mocker.AsyncMock(return_value=None)

    # Return the cache mock directly as an async mock.
    mock_cache_constructor = mocker.patch(
        'aif_gen.generate.caching.AsyncElasticsearchCache.maybe_from_env_var',
        new_callable=mocker.AsyncMock,
        return_value=mock_cache,
    )

    continual_dataset = await filter_continual_alignment_dataset_style_normalize(
        mock_dataset, mock_model, mock_client, mock_semaphore
    )

    # Verify cache was used
    assert mock_cache_constructor.called
    assert mock_cache.get.call_count >= 1
    assert mock_cache.set.call_count >= 1
    assert mock_cache.close.call_count == 1
