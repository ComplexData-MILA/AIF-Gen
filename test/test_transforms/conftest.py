import pytest

from aif_gen.util.seed import seed_everything


@pytest.fixture(autouse=True)
def run_seed_before_tests():
    seed_everything(1)
    yield


@pytest.fixture(params=[True, False])
def in_place(request):
    return request.param


@pytest.fixture(params=['call', 'apply', 'functional'])
def application_type(request):
    return request.param


def mock_task():
    return {
        'domain': {
            'Component A': {
                'name': 'Component A',
                'seed_words': ['a_foo', 'a_bar', 'a_baz'],
                'description': None,
                'weight': 0.5,
            },
            'Component B': {
                'name': 'Component B',
                'seed_words': ['b_foo', 'b_bar'],
                'description': None,
                'weight': 0.5,
            },
        },
        'objective': 'Mock Objective 1',
        'preference': 'Mock Preference 1',
    }
