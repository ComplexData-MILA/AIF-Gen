import asyncio
import json

import pytest

from aif_gen.dataset import (
    AlignmentDataset,
    AlignmentDatasetSample,
    ContinualAlignmentDataset,
)
from aif_gen.dataset.validation.llm_judge import llm_judge_validation
from aif_gen.task.alignment_task import AlignmentTask
from aif_gen.task.domain import Domain

_EXP_KEYS = [
    'alignment_max',
    'alignment_mean',
    'alignment_median',
    'alignment_min',
    'coherence_chosen_max',
    'coherence_chosen_mean',
    'coherence_chosen_median',
    'coherence_chosen_min',
    'coherence_rejected_max',
    'coherence_rejected_mean',
    'coherence_rejected_median',
    'coherence_rejected_min',
]


def async_mock_return(result):
    fut = asyncio.Future()
    fut.set_result(result)
    return fut


@pytest.fixture(params=[-1, 0, 1, 5, 10, 12])  # Changed to integers
def mock_client(mocker, request):
    mock_response = mocker.MagicMock(name='response')
    mock_response.choices[0].message.content = json.dumps(
        {
            'score': request.param,
        }
    )
    mock_client = mocker.MagicMock(name='client')
    mock_client.chat.completions.create.return_value = async_mock_return(mock_response)
    mock_client.score = request.param
    return mock_client


@pytest.fixture
def mock_client_uncaught_exception(mocker):
    mock_response = mocker.MagicMock(name='response')
    mock_response.choices[0].message.content = json.dumps(
        {
            'score': 5,  # Changed to integer
        }
    )
    mock_client = mocker.MagicMock(name='client')
    # Some uncaught exception (e.g. openai.NotFound)
    mock_client.chat.completions.create.side_effect = Exception
    mock_client.score = 5  # Changed to integer
    return mock_client


@pytest.fixture(
    params=[
        {'score': 'Bad response'},  # 'score' is not an integer
        {'foo': 'Mock response'},  # Missing the 'score'
    ]
)
def mock_client_schema_parse_exception(mocker, request):
    # Fail the first sample schema parse, then switch to good schema, and ensure the other samples are generated
    mock_bad_response = mocker.MagicMock(name='response')
    mock_bad_response.choices[0].message.content = json.dumps(request.param)

    mock_good_response = mocker.MagicMock(name='response')
    mock_good_response.choices[0].message.content = json.dumps(
        {
            'score': 5,  # Changed to integer
        }
    )

    mock_client = mocker.MagicMock(name='client')

    # This is a crude way of making the mock return a 'bad response' the first 5 times,
    # then 'good response' for the next 100 calls. 100 is arbitrary, but it should be
    # larger than the mock dataset we test here. If we do run on more than 100 samples,
    # this will raise a StopIteration exception, causes a flaky test. The proper way to
    # mock this would involve looking at the calling args, since the first API call doesn't
    # actually check for json schema.
    mock_client.chat.completions.create.side_effect = 5 * [
        async_mock_return(mock_bad_response)
    ] + 100 * [async_mock_return(mock_good_response)]

    mock_client.score = 5  # Changed to integer
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
def mock_task():
    return AlignmentTask(
        domain=Domain.from_dict({'education': {}}), objective='', preference=''
    )


@pytest.mark.asyncio
async def test_llm_judge_validation(mock_model, mock_client, mock_semaphore, mock_task):
    samples = [
        AlignmentDatasetSample(
            'Mock prompt A', 'Winning Response A 1', 'Losing Response A 1'
        ),
    ]
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    result = await llm_judge_validation(
        dataset, mock_model, mock_client, mock_semaphore
    )
    assert isinstance(result, list)
    assert sorted(list(result[0].keys())) == sorted(_EXP_KEYS)
    for v in result[0].values():
        assert v == max(0, min(10, mock_client.score))


@pytest.mark.asyncio
async def test_llm_judge_validation_empty_dataset(
    mock_model, mock_client, mock_semaphore, mock_task
):
    dataset = AlignmentDataset(task=mock_task, samples=[])
    result = await llm_judge_validation(
        dataset, mock_model, mock_client, mock_semaphore
    )
    assert result == [None]


@pytest.mark.asyncio
async def test_llm_judge_validation_continual_dataset(
    mock_model, mock_client, mock_semaphore, mock_task
):
    samples = [
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response A 1', 'Losing Response A 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response B 1', 'Losing Response B 1'
        ),
        AlignmentDatasetSample(
            'Mock prompt', 'Winning Response C 1', 'Losing Response C 1'
        ),
    ]
    dataset1 = AlignmentDataset(task=mock_task, samples=samples)
    dataset2 = AlignmentDataset(task=mock_task, samples=[])
    dataset = ContinualAlignmentDataset(datasets=[dataset1, dataset2])

    result = await llm_judge_validation(
        dataset, mock_model, mock_client, mock_semaphore
    )

    assert isinstance(result, list)
    assert sorted(list(result[0].keys())) == sorted(_EXP_KEYS)
    for v in result[0].values():
        assert v == max(0, min(10, mock_client.score))
    assert result[1] == None


@pytest.mark.asyncio
async def test_llm_judge_validation_with_parse_failures(
    mock_model, mock_client_schema_parse_exception, mock_semaphore, mock_task
):
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
    dataset = AlignmentDataset(task=mock_task, samples=samples)

    result = await llm_judge_validation(
        dataset, mock_model, mock_client_schema_parse_exception, mock_semaphore
    )

    assert isinstance(result, list)
    assert sorted(list(result[0].keys())) == sorted(_EXP_KEYS)
    for v in result[0].values():
        assert v == max(0, min(10, mock_client_schema_parse_exception.score))


@pytest.mark.asyncio
async def test_llm_judge_validation_uncaught_exception(
    mock_model, mock_client_uncaught_exception, mock_semaphore, mock_task
):
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
    dataset = AlignmentDataset(task=mock_task, samples=samples)
    with pytest.raises(Exception):
        results = await llm_judge_validation(
            dataset, mock_model, mock_client_uncaught_exception, mock_semaphore
        )
        assert results is None
