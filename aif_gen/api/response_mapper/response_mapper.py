import random
from textwrap import dedent
from typing import Optional

from aif_gen.task import AlignmentTask

from .base import ResponseMapperBase


class ResponseMapper(ResponseMapperBase):
    r"""Generate a prompt that, when given to a language model, produces a winning and losing response to the task_prompt.

    Args:
        suffix_context (Optional[str]=None): Optionally add arbitrary context at the end of the generated prompt.
    """

    NUMBER_OF_PREFERENCE_AXES_SAMPLED: int = 3
    TASK_PREFERENCE_INCLUSION_PROBABILITY_POSIIVE: float = 0.4
    TASK_PREFERENCE_INCLUSION_PROBABILITY_NEGATIVE: float = 0.4

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
        Generate a 'chosen' and 'rejected' response pair to the following prompt: '{task_prompt}'.\n

        The 'chosen' response should respond to the prompt according to the following preference: '{task.preference}'.
        The 'rejected' response should still respond to the prompt according to the preference but negligibly worse in its quality, however still close to the chosen response so it confuses the reader which one is actually better. Consider exactly the same style and lengths for the chosen and rejected please.
        You don't need to start your prompt by saying 'User asks' or start by "chosen:" or "rejected:".
        """
        if self.suffix_context:
            prompt += self.suffix_context

        prompt = dedent(prompt)
        return prompt

    def generate_no_preference_prompt(
        self, task: AlignmentTask, task_prompt: str, scores: list[int], parity: int = 0
    ) -> str:
        """Generate prompt without explicitly stating the user preference."""
        prob1 = random.random()
        prob2 = random.random()
        description_positive = ''
        description_negative = ''
        if prob1 < self.TASK_PREFERENCE_INCLUSION_PROBABILITY_POSIIVE and parity == 0:
            description_positive = f'Your response must follow this preference overall: PREFERENCE: {task.preference}: \n'
        elif (
            prob2 < self.TASK_PREFERENCE_INCLUSION_PROBABILITY_NEGATIVE and parity == 1
        ):
            description_negative = f'Your response must not respect this preference overall: PREFERENCE: {task.preference}: \n'
        prompt = f"""\
        Generate a response to the following prompt: '{task_prompt}'.\\n
        {description_positive}{description_negative}
        and
        {self.preference_axes_scale(scores)}
        You don't need to start your response by saying "here is the response" nor to give any meta-explanation. Just provide the response.
        """
        if self.suffix_context:
            prompt += self.suffix_context

        prompt = dedent(prompt)
        return prompt

    @property
    def preference_axes(self) -> list[tuple[str, str]]:
        return self._preference_axes

    @property
    def suffix_context(self) -> Optional[str]:
        f"""Optional added suffix context into the generated prompt."""
        return self._suffix_context

    def preference_axes_scale(
        self, scores: list[int], min_score: int = 1, max_score: int = 2
    ) -> str:
        """Generate a guide string for the preference axes scale to paste into the response meta-prompt.

        Args:
            scores (list[int]): List of scores for the preference axes.
            min_score (int, optional): Minimum score for the preference axes scale. Defaults to 1.
            max_score (int, optional): Maximum score for the preference axes scale. Defaults to 5.

        Returns:
            str: Description of the preference axes scale.
        """
        description: str = ''
        sampled_axes = random.sample(
            self.preference_axes, self.NUMBER_OF_PREFERENCE_AXES_SAMPLED
        )
        for i, axis in enumerate(sampled_axes):
            description += f'On a scale of {min_score} to {max_score} where {min_score} is {axis[0]} and {max_score} is {axis[1]}, your response should be: {scores[i]} \n'
        description += 'Please ensure your responses align with the provided scores.'
        return description
