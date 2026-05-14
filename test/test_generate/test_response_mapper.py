from aif_gen.generate.mappers import (
    PERSONA_ALIGNED,
    PERSONA_ANTI_ALIGNED,
    PERSONA_NEUTRAL,
    VALID_PERSONAS,
    ResponseMapper,
)
from aif_gen.task import AlignmentTask, Domain, DomainComponent


def test_init():
    mapper = ResponseMapper()
    assert mapper.suffix_context is None

    mapper = ResponseMapper(suffix_context='foo')
    assert mapper.suffix_context == 'foo'


def test_generate_response(suffix_context):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    domain = Domain(components=[health_component])
    preference = 'Generate responses that are vividly descriptive and engaging.'
    task = AlignmentTask(domain=domain, objective='mock', preference=preference)

    response_mapper = ResponseMapper(suffix_context=suffix_context)
    task_prompt = 'Create a story about how the rise of medicine could make exercise no longer necessary.'
    prompt = response_mapper.generate_prompt(task, task_prompt)

    assert preference in prompt
    if suffix_context is not None:
        assert suffix_context in prompt


def test_generate_no_preference_response(suffix_context):
    health_component = DomainComponent(
        name='Health', seed_words=['hospital', 'medicine', 'exercise']
    )
    domain = Domain(components=[health_component])
    preference = 'Generate responses that are vividly descriptive and engaging.'
    task = AlignmentTask(domain=domain, objective='mock', preference=preference)

    task_prompt = 'Create a story about how the rise of medicine could make exercise no longer necessary.'
    response_mapper = ResponseMapper(suffix_context=suffix_context)
    prompt1, prompt2 = response_mapper.generate_no_preference_prompt(task, task_prompt)

    for prompt in [prompt1, prompt2]:
        scale_lines = [
            ln for ln in prompt.splitlines() if ln.strip().startswith('On a scale')
        ]
        # you should have exactly NUM_PREFERENCE_AXES_SAMPLES of those
        assert len(scale_lines) == response_mapper.NUM_PREFERENCE_AXES_SAMPLES

        # suffix_context still shows up if present
        if suffix_context:
            assert suffix_context in prompt


def _mk_task() -> AlignmentTask:
    domain = Domain(components=[DomainComponent(name='Health', seed_words=['x'])])
    return AlignmentTask(
        domain=domain,
        objective='mock',
        preference='Generate responses that are vividly descriptive and engaging.',
    )


def test_generate_candidate_prompt_aligned_includes_preference():
    mapper = ResponseMapper()
    task = _mk_task()
    p = mapper.generate_candidate_prompt(task, 'tell a story', PERSONA_ALIGNED)
    assert task.preference in p
    assert 'follow' in p.lower()


def test_generate_candidate_prompt_anti_aligned_includes_violation_clause():
    mapper = ResponseMapper()
    task = _mk_task()
    p = mapper.generate_candidate_prompt(task, 'tell a story', PERSONA_ANTI_ALIGNED)
    assert task.preference in p
    assert 'violate' in p.lower() or 'deliberately' in p.lower()


def test_generate_candidate_prompt_neutral_makes_no_preference_claim():
    mapper = ResponseMapper()
    task = _mk_task()
    p = mapper.generate_candidate_prompt(task, 'tell a story', PERSONA_NEUTRAL)
    # Neutral prompt must NOT instruct following the preference, and must NOT
    # cite the task preference text.
    assert task.preference not in p
    assert 'naturally' in p.lower() or 'without considering' in p.lower()


def test_generate_candidate_prompt_rejects_unknown_persona():
    import pytest

    mapper = ResponseMapper()
    task = _mk_task()
    with pytest.raises(ValueError):
        mapper.generate_candidate_prompt(task, 'tell a story', 'bogus')


def test_default_persona_schedule_length_and_coverage():
    mapper = ResponseMapper()
    schedule = mapper.default_persona_schedule(6, base_temperature=1.0)
    assert len(schedule) == 6
    personas = {p for p, _ in schedule}
    assert PERSONA_ALIGNED in personas
    assert PERSONA_ANTI_ALIGNED in personas
    for p, t in schedule:
        assert p in VALID_PERSONAS
        assert 0.0 <= t <= 2.0
