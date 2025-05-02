import pytest

from aif_gen.generate.mappers import PromptMapper
from aif_gen.generate.mappers.base import ETHICAL_GUIDELINES
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def test_init():
    mapper = PromptMapper()
    assert mapper.max_seed_word_samples == 2
    assert mapper.suffix_context is None

    mapper = PromptMapper(max_seed_word_samples=20, suffix_context='foo')
    assert mapper.max_seed_word_samples == 20
    assert mapper.suffix_context == 'foo'


@pytest.mark.parametrize('bad_max_seed_word_samples', [-1, 0])
def test_init_bad_max_seed_word_samples(bad_max_seed_word_samples):
    with pytest.raises(ValueError):
        _ = PromptMapper(bad_max_seed_word_samples)


def test_generate_prompt_uniform_component_weights(
    max_seed_word_samples, suffix_context
):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    tech_component = DomainComponent(
        name='Tech',
        seed_words=['technology', 'iPhone', 'Bluetooth', 'internet', 'chat-gpt'],
    )
    domain = Domain(components=[health_component, tech_component], weights=[3, 3])
    objective = 'Align LLM outputs with ethical healthcare practices.'
    task = AlignmentTask(domain=domain, objective=objective, preference='mock')
    all_seed_words = health_component.seed_words + tech_component.seed_words

    prompt_mapper = PromptMapper(max_seed_word_samples, suffix_context=suffix_context)
    prompt = prompt_mapper.generate_prompt(task)

    assert ETHICAL_GUIDELINES in prompt
    assert objective in prompt
    assert any(word in prompt for word in all_seed_words)
    if suffix_context is not None:
        assert suffix_context in prompt


def test_generate_prompt_non_uniform_component_weights(
    max_seed_word_samples, suffix_context
):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    tech_component = DomainComponent(
        name='Tech',
        seed_words=['technology', 'iPhone', 'Bluetooth', 'internet', 'chat-gpt'],
    )
    domain = Domain(components=[health_component, tech_component], weights=[30, 3])
    objective = 'Align LLM outputs with ethical healthcare practices.'
    task = AlignmentTask(domain=domain, objective=objective, preference='mock')
    all_seed_words = health_component.seed_words + tech_component.seed_words

    prompt_mapper = PromptMapper(max_seed_word_samples, suffix_context=suffix_context)
    prompt = prompt_mapper.generate_prompt(task)

    assert ETHICAL_GUIDELINES in prompt
    assert objective in prompt
    assert any(word in prompt for word in all_seed_words)
    if suffix_context is not None:
        assert suffix_context in prompt


def test_generate_prompt_single_domain_component(max_seed_word_samples, suffix_context):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    domain = Domain(components=[health_component])
    objective = 'Align LLM outputs with ethical healthcare practices.'
    task = AlignmentTask(domain=domain, objective=objective, preference='mock')
    all_seed_words = health_component.seed_words

    prompt_mapper = PromptMapper(max_seed_word_samples, suffix_context=suffix_context)
    prompt = prompt_mapper.generate_prompt(task)

    assert ETHICAL_GUIDELINES in prompt
    assert objective in prompt
    assert any(word in prompt for word in all_seed_words)
    if suffix_context is not None:
        assert suffix_context in prompt
