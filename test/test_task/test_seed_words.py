import pytest

from aif_gen.task.seed_words import get_seed_words


@pytest.mark.parametrize(
    'seed_word_alias', ['education', 'finance', 'healthcare', 'politics', 'technology']
)
def test_get_seed_words(seed_word_alias):
    seed_words = get_seed_words(seed_word_alias)
    assert isinstance(seed_words, list)
    assert len(seed_words) > 0
    assert all([isinstance(word, str) for word in seed_words])


@pytest.mark.parametrize('bad_seed_word_alias', ['foo', ''])
def test_get_seed_words_bad_alias(bad_seed_word_alias):
    with pytest.raises(ValueError):
        _ = get_seed_words(bad_seed_word_alias)
