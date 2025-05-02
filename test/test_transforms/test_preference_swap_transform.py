import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.transforms import PreferenceSwapTransform
from aif_gen.transforms import functional as F

from .conftest import mock_task


def mock_dataset_dict(continual):
    if continual:
        return {
            'datasets': [
                {
                    'task': mock_task(),
                    'train': [
                        {
                            'prompt': 'Mock prompt A 1',
                            'chosen': 'Winning Response A 1',
                            'rejected': 'Losing Response A 1',
                        },
                        {
                            'prompt': 'Mock prompt B 1',
                            'chosen': 'Winning Response B 1',
                            'rejected': 'Losing Response B 1',
                        },
                        {
                            'prompt': 'Mock prompt C 1',
                            'chosen': 'Winning Response C 1',
                            'rejected': 'Losing Response C 1',
                        },
                    ],
                    'test': [],
                },
                {
                    'task': mock_task(),
                    'train': [
                        {
                            'prompt': 'Mock prompt A 2',
                            'chosen': 'Winning Response A 2',
                            'rejected': 'Losing Response A 2',
                        },
                        {
                            'prompt': 'Mock prompt B 2',
                            'chosen': 'Winning Response B 2',
                            'rejected': 'Losing Response B 2',
                        },
                        {
                            'prompt': 'Mock prompt C 2',
                            'chosen': 'Winning Response C 2',
                            'rejected': 'Losing Response C 2',
                        },
                    ],
                    'test': [],
                },
            ]
        }
    else:
        return {
            'task': mock_task(),
            'train': [
                {
                    'prompt': 'Mock prompt A 1',
                    'chosen': 'Winning Response A 1',
                    'rejected': 'Losing Response A 1',
                },
                {
                    'prompt': 'Mock prompt B 1',
                    'chosen': 'Winning Response B 1',
                    'rejected': 'Losing Response B 1',
                },
                {
                    'prompt': 'Mock prompt C 1',
                    'chosen': 'Winning Response C 1',
                    'rejected': 'Losing Response C 1',
                },
            ],
            'test': [],
        }


@pytest.mark.parametrize('bad_swap_probability', [-1, 2])
def test_apply_preference_swap_bad_swap_probability(bad_swap_probability):
    with pytest.raises(ValueError):
        _ = PreferenceSwapTransform(swap_probability=bad_swap_probability)

    # Check failure when using the setter method
    transform = PreferenceSwapTransform(swap_probability=0.3)
    with pytest.raises(ValueError):
        transform.swap_probability = bad_swap_probability

    # Check failure when using the functional API
    mock_dataset = None
    with pytest.raises(ValueError):
        F.preference_swap_transform(mock_dataset, swap_probability=bad_swap_probability)


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=False)])
def test_apply_preference_swap_to_static_dataset(
    dataset_dict, in_place, application_type
):
    expected_dataset_dict = {
        'task': mock_task(),
        'train': [
            {
                'prompt': 'Mock prompt A 1',
                'chosen': 'Winning Response A 1',
                'rejected': 'Losing Response A 1',
            },
            {
                'prompt': 'Mock prompt B 1',
                'chosen': 'Losing Response B 1',
                'rejected': 'Winning Response B 1',
            },
            {
                'prompt': 'Mock prompt C 1',
                'chosen': 'Winning Response C 1',
                'rejected': 'Losing Response C 1',
            },
        ],
        'test': [],
    }

    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=0.5, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=False)])
def test_apply_preference_swap_to_static_dataset_full_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = {
        'task': mock_task(),
        'train': [
            {
                'prompt': 'Mock prompt A 1',
                'chosen': 'Losing Response A 1',
                'rejected': 'Winning Response A 1',
            },
            {
                'prompt': 'Mock prompt B 1',
                'chosen': 'Losing Response B 1',
                'rejected': 'Winning Response B 1',
            },
            {
                'prompt': 'Mock prompt C 1',
                'chosen': 'Losing Response C 1',
                'rejected': 'Winning Response C 1',
            },
        ],
        'test': [],
    }

    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 1  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=1, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=True)])
def test_apply_preference_swap_to_continual_dataset_full_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = {
        'datasets': [
            {
                'task': mock_task(),
                'train': [
                    {
                        'prompt': 'Mock prompt A 1',
                        'chosen': 'Losing Response A 1',
                        'rejected': 'Winning Response A 1',
                    },
                    {
                        'prompt': 'Mock prompt B 1',
                        'chosen': 'Losing Response B 1',
                        'rejected': 'Winning Response B 1',
                    },
                    {
                        'prompt': 'Mock prompt C 1',
                        'chosen': 'Losing Response C 1',
                        'rejected': 'Winning Response C 1',
                    },
                ],
                'test': [],
            },
            {
                'task': mock_task(),
                'train': [
                    {
                        'prompt': 'Mock prompt A 2',
                        'chosen': 'Losing Response A 2',
                        'rejected': 'Winning Response A 2',
                    },
                    {
                        'prompt': 'Mock prompt B 2',
                        'chosen': 'Losing Response B 2',
                        'rejected': 'Winning Response B 2',
                    },
                    {
                        'prompt': 'Mock prompt C 2',
                        'chosen': 'Losing Response C 2',
                        'rejected': 'Winning Response C 2',
                    },
                ],
                'test': [],
            },
        ]
    }

    dataset = ContinualAlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 1  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=1, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=False)])
def test_apply_preference_swap_to_static_dataset_no_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = dataset_dict
    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 0  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=0, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=True)])
def test_apply_preference_swap_to_continual_dataset_no_swap(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = dataset_dict
    dataset = ContinualAlignmentDataset.from_dict(dataset_dict)
    transform = PreferenceSwapTransform(0.5)
    transform.swap_probability = 0  # Setter overrides swap probability

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.preference_swap_transform(
            dataset, swap_probability=0, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset
