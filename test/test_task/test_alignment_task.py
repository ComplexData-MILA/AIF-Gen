import pytest

from aif_gen.task import AlignmentTask, Domain


def test_init():
    component_dict = {
        'Component A': {
            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
            'description': 'A Mock Domain Component',
        },
        'Component B': {
            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
            'description': 'B Mock Domain Component',
        },
    }
    domain = Domain.from_dict(component_dict)
    objective = 'Mock Objective'
    preference = 'Mock Preference'

    task = AlignmentTask(domain, objective, preference)
    exp_str = f'AlignmentTask({str(domain)}, Objective: Mock Objective, Preference: Mock Preference)'
    assert str(task) == exp_str


def test_init_from_dict():
    task_dict = {
        'domain': {
            'Component A': {
                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                'description': 'A Mock Domain Component',
            },
            'Component B': {
                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                'description': 'B Mock Domain Component',
            },
        },
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    task = AlignmentTask.from_dict(task_dict)
    domain = Domain.from_dict(task_dict['domain'])
    exp_str = f'AlignmentTask({str(domain)}, Objective: Mock Objective, Preference: Mock Preference)'
    assert str(task) == exp_str


def test_init_from_dict_missing_keys():
    task_dict = {  # Missing 'domain' key
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    with pytest.raises(ValueError):
        _ = AlignmentTask.from_dict(task_dict)


def test_to_dict_no_weights():
    task_dict = {
        'domain': {
            'Component A': {
                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                'description': 'A Mock Domain Component',
            },
            'Component B': {
                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                'description': 'B Mock Domain Component',
            },
        },
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    # Note: We automatically add uniform weights to the domain if they were not specified
    expected_dict = task_dict
    expected_dict['domain']['Component A']['weight'] = 0.5
    expected_dict['domain']['Component B']['weight'] = 0.5

    task = AlignmentTask.from_dict(task_dict)
    assert expected_dict == task.to_dict()


def test_to_dict_with_weights():
    task_dict = {
        'domain': {
            'Component A': {
                'weight': 0.3,
                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                'description': 'A Mock Domain Component',
            },
            'Component B': {
                'weight': 0.7,
                'seed_words': ['b_foo', 'b_bar', 'b_baz'],
                'description': 'B Mock Domain Component',
            },
        },
        'objective': 'Mock Objective',
        'preference': 'Mock Preference',
    }

    expected_dict = task_dict

    task = AlignmentTask.from_dict(task_dict)
    assert expected_dict == task.to_dict()
