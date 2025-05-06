import pytest

from aif_gen.dataset import AlignmentDataset, ContinualAlignmentDataset
from aif_gen.transforms import SplitTransform
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


@pytest.mark.parametrize('bad_test_ratio', [-1, 2])
def test_apply_split_transform_bad_test_ratio(bad_test_ratio):
    with pytest.raises(ValueError):
        _ = SplitTransform(test_ratio=bad_test_ratio)

    # Check failure when using the setter method
    transform = SplitTransform(test_ratio=0.3)
    with pytest.raises(ValueError):
        transform.test_ratio = bad_test_ratio

    # Check failure when using the functional API
    mock_dataset = None
    with pytest.raises(ValueError):
        F.split_transform(mock_dataset, test_ratio=bad_test_ratio)


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=False)])
def test_apply_split_transform_to_static_dataset(
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
        ],
        'test': [
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
    }

    dataset = AlignmentDataset.from_dict(dataset_dict)
    transform = SplitTransform(0.5)

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.split_transform(
            dataset, test_ratio=0.5, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset


@pytest.mark.parametrize('dataset_dict', [mock_dataset_dict(continual=True)])
def test_apply_split_transform_to_continual_dataset(
    in_place, application_type, dataset_dict
):
    expected_dataset_dict = {
        'datasets': [
            {
                'task': mock_task(),
                'train': [
                    {
                        'prompt': 'Mock prompt A 1',
                        'chosen': 'Winning Response A 1',
                        'rejected': 'Losing Response A 1',
                    },
                ],
                'test': [
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
            },
            {
                'task': mock_task(),
                'train': [
                    {
                        'prompt': 'Mock prompt A 2',
                        'chosen': 'Winning Response A 2',
                        'rejected': 'Losing Response A 2',
                    },
                ],
                'test': [
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
            },
        ]
    }

    dataset = ContinualAlignmentDataset.from_dict(dataset_dict)
    transform = SplitTransform(0.5)

    transformed_dataset = None
    if application_type == 'call':
        transformed_dataset = transform(dataset, in_place)
    elif application_type == 'apply':
        transformed_dataset = transform.apply(dataset, in_place)
    elif application_type == 'functional':
        transformed_dataset = F.split_transform(
            dataset, test_ratio=0.5, in_place=in_place
        )

    assert transformed_dataset.to_dict() == expected_dataset_dict
    if in_place:
        assert dataset == transformed_dataset
