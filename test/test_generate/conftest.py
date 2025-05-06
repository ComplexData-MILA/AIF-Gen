import pytest

from aif_gen.util.seed import seed_everything


@pytest.fixture(autouse=True)
def run_seed_before_tests():
    seed_everything(1)
    yield


@pytest.fixture(params=[None, 'Mock suffix context to add to the prompt'])
def suffix_context(request):
    return request.param


@pytest.fixture(params=[1, 10, 20])
def max_seed_word_samples(request):
    return request.param
