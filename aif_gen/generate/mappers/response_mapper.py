import random
from textwrap import dedent
from typing import Optional, Tuple

from aif_gen.task import AlignmentTask

from .base import ResponseMapperBase


class ResponseMapper(ResponseMapperBase):
    r"""Generate a prompt that, when given to a language model, produces a winning and losing response to the task_prompt.

    Args:
        suffix_context (Optional[str]=None): Optional suffix text to add at the end of the generated prompt.
    """

    NUM_PREFERENCE_AXES_SAMPLES: int = 3
    PREFERENCE_INCLUSION_PROB_POS: float = 0.5
    PREFERENCE_INCLUSION_PROB_NEG: float = 0.5

    def __init__(self, suffix_context: Optional[str] = None) -> None:
        self._suffix_context = suffix_context
        self._preference_axes = [
            ('short', 'long'),
            ('formal', 'casual'),
            ('helpful', 'harmful'),
            ('expert', 'eli5'),
            ('direct', 'hinted'),
            ('authoritative', 'tentative'),
            ('friendly', 'distance'),
            ('optimistic', 'pessimistic'),
            ('serious', 'humorous'),
            ('respectful', 'disrespectful'),
            ('complex', 'simple'),
            ('neutral', 'biased'),
            ('detailed', 'abstract'),
        ]  # TODO could be added to the config - or finalized

    def generate_prompt(self, task: AlignmentTask, task_prompt: str) -> str:
        prompt = f"""\
        Generate a 'chosen' and 'rejected' response pair to the following prompt: '{task_prompt}'.
        The 'chosen' response should respond to the prompt according to the following preference: '{task.preference}'.
        The 'rejected' response should still respond to the prompt according to the preference but negligibly worse in its quality,
        however still close to the chosen response so it confuses the reader which one is actually better.
        Consider exactly the same style and lengths for the chosen and rejected please.
        You don't need to start your response by saying "here is the response" nor to give any meta-explanation. Just provide the response.
        """
        if self.suffix_context:
            prompt += self.suffix_context
        return dedent(prompt)

    def generate_no_preference_prompt(
        self, task: AlignmentTask, task_prompt: str
    ) -> Tuple[str, str]:
        scores = [random.randint(1, 5) for _ in range(self.NUM_PREFERENCE_AXES_SAMPLES)]

        def _generate_no_preference_prompt(parity: int) -> str:
            desc_pos, desc_neg = '', ''
            if random.random() < self.PREFERENCE_INCLUSION_PROB_POS and parity == 0:
                desc_pos = f'Your response must follow this preference overall: PREFERENCE: {task.preference}\n'
            elif random.random() < self.PREFERENCE_INCLUSION_PROB_NEG and parity == 1:
                desc_neg = f'Your response must not respect this preference overall: PREFERENCE: {task.preference}\n'

            prompt = f"""\
            Generate a response to the following prompt: '{task_prompt}'.
            {desc_pos}{desc_neg} and
            {self._preference_axes_scale(scores)}'
            You don't need to start your response by saying "here is the response" nor to give any meta-explanation. Just provide the response.
            """
            if self.suffix_context:
                prompt += self.suffix_context
            return dedent(prompt)

        prompt1 = _generate_no_preference_prompt(parity=0)
        prompt2 = _generate_no_preference_prompt(parity=1)
        return prompt1, prompt2

    @property
    def preference_axes(self) -> list[tuple[str, str]]:
        return self._preference_axes

    @property
    def suffix_context(self) -> Optional[str]:
        f"""Optional added suffix context into the generated prompt."""
        return self._suffix_context

    def _preference_axes_scale(
        self, scores: list[int], min_score: int = 1, max_score: int = 2
    ) -> str:
        axes = random.sample(self.preference_axes, self.NUM_PREFERENCE_AXES_SAMPLES)
        desc = ''
        for i, axis in enumerate(axes):
            desc += f'On a scale of {min_score} to {max_score} where {min_score} is {axis[0]} and {max_score} is {axis[1]}, your response should be: {scores[i]}\n'
        desc += 'Please ensure your responses aligns with the provided scores.'
        return desc
