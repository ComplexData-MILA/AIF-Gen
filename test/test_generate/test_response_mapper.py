import random

from aif_gen.generate.mappers import ResponseMapper
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
    # build a scores list at least as long as NUM_PREFERENCE_AXES_SAMPLES
    scores = [
        random.randint(1, 5) for _ in range(response_mapper.NUM_PREFERENCE_AXES_SAMPLES)
    ]
    prompt = response_mapper.generate_no_preference_prompt(task, task_prompt, scores)

    # grab only the "On a scale..." lines
    scale_lines = [
        ln for ln in prompt.splitlines() if ln.strip().startswith('On a scale')
    ]
    # you should have exactly NUM_PREFERENCE_AXES_SAMPLES of those
    assert len(scale_lines) == response_mapper.NUM_PREFERENCE_AXES_SAMPLES

    # each line must contain the score you passed in, in order
    for idx, ln in enumerate(scale_lines):
        assert str(scores[idx]) in ln

    # suffix_context still shows up if present
    if suffix_context:
        assert suffix_context in prompt
