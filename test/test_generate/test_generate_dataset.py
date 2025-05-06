import asyncio
import itertools
import json

import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.dataset.alignment_sample import AlignmentDatasetSample
from aif_gen.generate import generate_continual_dataset
from aif_gen.task import AlignmentTask


def async_mock_return(result):
    fut = asyncio.Future()
    fut.set_result(result)
    return fut


def async_mock_coro(result):
    async def _coro():
        return result

    return _coro()


@pytest.fixture(params=[1.0, 0.0])
def mock_score(monkeypatch, request):
    """Stub _get_score to return request.param; controls flipping."""
    score = request.param

    async def fake_get_alignment_score(*args, **kwargs):
        return score, kwargs.get('dataset_idx'), kwargs.get('metric_name')

    monkeypatch.setattr(
        'aif_gen.validation.llm_judge._get_score',
        fake_get_alignment_score,
    )
    return score


@pytest.fixture
def mock_client_no_preference_axis(mocker):
    mock_prompt_response = mocker.MagicMock(name='response')
    mock_prompt_response.choices[0].message.content = json.dumps(
        {'prompt': 'Mock prompt'}
    )

    mock_chosen_rejected_response = mocker.MagicMock(name='response')
    mock_chosen_rejected_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'Mock chosen response',
            'rejected': 'Mock rejected response',
        }
    )

    # Crude way of ensuring that every time we call 'chat_completion',
    # the mock client switches between a valid prompt_response schema
    # and a valid chosen_rejected_response schema.
    mock_client = mocker.MagicMock(name='client')
    mock_client.chat.completions.create.side_effect = [
        async_mock_return(mock_prompt_response),
        async_mock_return(mock_chosen_rejected_response),
    ] * 100

    return mock_client


@pytest.fixture
def mock_client_with_preference_axes(mocker):
    mock_prompt_response = mocker.MagicMock(name='prompt_response')
    mock_prompt_response.choices[0].message.content = json.dumps(
        {'prompt': 'Mock prompt with preference axes'}
    )

    mock_response1 = mocker.MagicMock(name='response1')
    mock_response1.choices[0].message.content = json.dumps(
        {'response': 'Mock response 1'}
    )
    mock_response2 = mocker.MagicMock(name='response2')
    mock_response2.choices[0].message.content = json.dumps(
        {'response': 'Mock response 2'}
    )

    mock_client = mocker.MagicMock(name='client')

    # build a cycle of factories
    factories = itertools.cycle(
        [
            lambda: async_mock_coro(mock_prompt_response),
            lambda: async_mock_coro(mock_response1),
            lambda: async_mock_coro(mock_response2),
        ]
    )
    mock_client.chat.completions.create.side_effect = lambda *args, **kwargs: next(
        factories
    )()
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


@pytest.fixture(
    params=[
        {'chosen': 'Mock chosen response'},  # Missing the 'rejected' response
        {'rejected': 'Mock rejected response'},  # Missing the 'chosen' response
        {'Foo Bar': 'Baz'},  # 'Missing both 'chosen' and 'rejected' keys
        'Foo Bar Baz',  # Not even a dictionary
    ]
)
def mock_client_schema_parse_exception(mocker, request):
    # Fail the first sample schema parse, then switch to good schema, and ensure the other samples are generated
    mock_bad_chosen_rejected_response = mocker.MagicMock(name='response')
    mock_bad_chosen_rejected_response.choices[0].message.content = json.dumps(
        request.param
    )

    mock_prompt_response = mocker.MagicMock(name='response')
    mock_prompt_response.choices[0].message.content = json.dumps(
        {'prompt': 'Mock prompt'}
    )
    mock_good_chosen_rejected_response = mocker.MagicMock(name='response')
    mock_good_chosen_rejected_response.choices[0].message.content = json.dumps(
        {
            'chosen': 'Mock chosen response',
            'rejected': 'Mock rejected response',
        }
    )

    mock_client = mocker.MagicMock(name='client')

    # This is a crude way of making the mock return a 'bad response' the first time,
    # then 'good response' for the next 100 calls. 100 is arbitrary, but it should be
    # larger than the mock dataset we test here. If we do run on more than 100 samples,
    # this will raise a StopIteration exception, causes a flaky test. The proper way to
    # mock this would involve looking at the calling args, since the first API call doesn't
    # actually check for json schema.
    mock_client.chat.completions.create.side_effect = [
        async_mock_return(mock_prompt_response),
        async_mock_return(mock_bad_chosen_rejected_response),
    ] + 100 * [
        async_mock_return(mock_prompt_response),
        async_mock_return(mock_good_chosen_rejected_response),
    ]

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
def mock_data_config():
    return {
        'task_specs': [
            {
                'num_samples': 5,
                'alignment_task': {
                    'domain': {
                        'Component A': {
                            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                        },
                        'Component B': {
                            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                        },
                    },
                    'objective': 'Mock Objective 1',
                    'preference': 'Mock Preference 1',
                },
            },
            {
                'num_samples': 5,
                'alignment_task': {
                    'domain': {
                        'Component A': {
                            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                        },
                        'Component B': {
                            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                        },
                    },
                    'objective': 'Mock Objective 2',
                    'preference': 'Mock Preference 2',
                },
            },
        ]
    }


@pytest.mark.asyncio
async def test_generate_continual_dataset_schema_parse_exception(
    mock_data_config, mock_model, mock_client_schema_parse_exception, mock_semaphore
):
    continual_dataset = await generate_continual_dataset(
        mock_data_config, mock_model, mock_client_schema_parse_exception, mock_semaphore
    )

    task_specs = mock_data_config['task_specs']
    assert isinstance(continual_dataset, ContinualAlignmentDataset)
    assert len(task_specs) == len(continual_dataset.datasets)

    for i in range(len(continual_dataset.datasets)):
        dataset = continual_dataset.datasets[i]
        assert isinstance(dataset, AlignmentDataset)

        exp_task = AlignmentTask.from_dict(task_specs[i]['alignment_task'])

        assert len(dataset) > 0  # TODO: This is flaky
        assert dataset.task.to_dict() == exp_task.to_dict()

        for sample in dataset.samples:
            assert isinstance(sample, AlignmentDatasetSample)


@pytest.mark.asyncio
async def test_generate_continual_dataset_uncaught_exception(
    mock_data_config, mock_model, mock_client_uncaught_exception, mock_semaphore
):
    with pytest.raises(Exception):
        continual_dataset = await generate_continual_dataset(
            mock_data_config, mock_model, mock_client_uncaught_exception, mock_semaphore
        )
        assert continual_dataset is None


@pytest.mark.asyncio
@pytest.mark.parametrize('pop_key', ['task_specs'])
async def test_generate_continual_dataset_bad_config_dict(
    mock_data_config,
    mock_model,
    mock_client_no_preference_axis,
    mock_semaphore,
    pop_key,
):
    mock_data_config.pop(pop_key)  # Remove required key
    with pytest.raises(KeyError):
        continual_dataset = await generate_continual_dataset(
            mock_data_config, mock_model, mock_client_no_preference_axis, mock_semaphore
        )
        assert continual_dataset is None
        assert (
            not mock_client_no_preference_axis.called
        )  # Make sure we didn't hit the model endpoint


@pytest.mark.asyncio
async def test_generate_continual_dataset_no_preference_axis(
    mock_data_config, mock_model, mock_client_no_preference_axis, mock_semaphore
):
    continual_dataset = await generate_continual_dataset(
        mock_data_config, mock_model, mock_client_no_preference_axis, mock_semaphore
    )
    task_specs = mock_data_config['task_specs']
    assert isinstance(continual_dataset, ContinualAlignmentDataset)
    assert len(task_specs) == len(continual_dataset.datasets)

    for i in range(len(continual_dataset.datasets)):
        dataset = continual_dataset.datasets[i]
        assert isinstance(dataset, AlignmentDataset)

        exp_num_samples = task_specs[i]['num_samples']
        exp_task = AlignmentTask.from_dict(task_specs[i]['alignment_task'])

        assert len(dataset) == exp_num_samples
        assert dataset.task.to_dict() == exp_task.to_dict()

        for sample in dataset.samples:
            assert isinstance(sample, AlignmentDatasetSample)


@pytest.mark.asyncio
async def test_generate_continual_dataset_with_preference_axes(
    mock_data_config,
    mock_model,
    mock_client_with_preference_axes,
    mock_semaphore,
    mock_score,  # yields 1.0 (no flip) or 0.0 (flip)
):
    """When score==1.0 no swap; when score==0.0 we swap chosen/rejected."""
    # force one sample per dataset
    for spec in mock_data_config['task_specs']:
        spec['num_samples'] = 1
    continual_dataset = await generate_continual_dataset(
        mock_data_config,
        mock_model,
        mock_client_with_preference_axes,
        mock_semaphore,
        include_preference_axes=True,
    )

    task_specs = mock_data_config['task_specs']
    assert isinstance(continual_dataset, ContinualAlignmentDataset)
    assert len(task_specs) == len(continual_dataset.datasets)

    for idx, dataset in enumerate(continual_dataset.datasets):
        assert isinstance(dataset, AlignmentDataset)
        # check the task is unchanged
        exp_task = AlignmentTask.from_dict(task_specs[idx]['alignment_task'])
        assert dataset.task.to_dict() == exp_task.to_dict()
        for sample in dataset.samples:
            assert isinstance(sample, AlignmentDatasetSample)
            if mock_score == 0.0:
                # judge says response2 wins → swap
                assert sample.chosen == 'Mock response 2'
                assert sample.rejected == 'Mock response 1'
            else:
                # no swap
                assert sample.chosen == 'Mock response 1'
                assert sample.rejected == 'Mock response 2'


@pytest.fixture
def mock_client_preference_axes_schema_parse_exception(mocker):
    # first prompt ok
    mock_prompt = mocker.MagicMock(name='prompt')
    mock_prompt.choices[0].message.content = json.dumps({'prompt': 'Bad→Good'})
    # bad single
    mock_bad = mocker.MagicMock(name='bad')
    mock_bad.choices[0].message.content = json.dumps({'foo': 'bar'})
    # good single
    mock_good = mocker.MagicMock(name='good')
    mock_good.choices[0].message.content = json.dumps({'response': 'OK'})
    # subsequent good prompts
    mock_good_prompt = mocker.MagicMock(name='good_prompt')
    mock_good_prompt.choices[0].message.content = json.dumps({'prompt': 'Good'})

    mock_client = mocker.MagicMock(name='client')
    # sequence of factories: prompt→bad→good, then repeating good_prompt→good→good
    seq_factories = [
        lambda: async_mock_coro(mock_prompt),
        lambda: async_mock_coro(mock_bad),
        lambda: async_mock_coro(mock_good),
    ] + list(
        itertools.islice(
            itertools.cycle(
                [
                    lambda: async_mock_coro(mock_good_prompt),
                    lambda: async_mock_coro(mock_good),
                    lambda: async_mock_coro(mock_good),
                ]
            ),
            0,
            300,
        )
    )
    seq_iter = iter(seq_factories)
    mock_client.chat.completions.create.side_effect = lambda *args, **kwargs: next(
        seq_iter
    )()
    return mock_client


@pytest.mark.asyncio
async def test_generate_continual_dataset_preference_axes_schema_parse_exception(
    mock_data_config,
    mock_model,
    mock_client_preference_axes_schema_parse_exception,
    mock_semaphore,
):
    # Set num_samples to 1 for this test to make expectations clearer
    for task_spec in mock_data_config['task_specs']:
        task_spec['num_samples'] = 1
    task_specs = mock_data_config['task_specs']
    continual_dataset = await generate_continual_dataset(
        mock_data_config,
        mock_model,
        mock_client_preference_axes_schema_parse_exception,
        mock_semaphore,
        include_preference_axes=True,
    )

    # Even though there was a schema parse error for one sample, the function should
    # continue and attempt to generate other samples
    assert isinstance(continual_dataset, ContinualAlignmentDataset)
    assert len(continual_dataset.datasets) == len(mock_data_config['task_specs'])

    for i in range(len(continual_dataset.datasets)):
        dataset = continual_dataset.datasets[i]
        assert isinstance(dataset, AlignmentDataset)

        exp_task = AlignmentTask.from_dict(task_specs[i]['alignment_task'])

        assert len(dataset) <= 1  # TODO: This is flaky
        assert dataset.task.to_dict() == exp_task.to_dict()

        for sample in dataset.samples:
            assert isinstance(sample, AlignmentDatasetSample)
