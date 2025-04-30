from textwrap import dedent
from typing import List, Optional

import numpy as np

from aif_gen.task import AlignmentTask

from .base import PromptMapperBase


class PromptMapper(PromptMapperBase):
    r"""Generate a prompt that, when given to a language model, produces a prompt for a given AlignmentTask.

    Samples domain component seed words (without replacement) from the AlignmentTask to contextualize the prompt.
    The sampling is parameterized by the weight of each component of the domain.

    Args:
        max_seed_word_samples (int): Maximum number of seed words to sample across all domain components (default=2)
        suffix_context (Optional[str]=None): Optionally add arbitrary context at the end of the generated prompt.
    """

    def __init__(
        self, max_seed_word_samples: int = 2, suffix_context: Optional[str] = None
    ) -> None:
        if max_seed_word_samples <= 0:
            raise ValueError(
                f'Max seed word samples must be positive, got: {max_seed_word_samples}'
            )

        self._max_seed_word_samples = max_seed_word_samples
        self._suffix_context = suffix_context

    def generate_prompt(self, task: AlignmentTask) -> str:
        seed_words = self._sample_seed_words(task)
        prompt = f"""\
        Generate a text that fulfills the objective below.
        Do exactly what the objective says: "{task.objective}".
        The description must include the following seed words: {','.join(seed_words)}.
        Do not include any meta commentary, instructions, or extra text (e.g., avoid phrases like "User asks" or additional context).
        The output should be clear and self-contained.
        You don't need to start by saying "prompt:".
        {self.ETHICAL_GUIDELINES}
        """
        if self.suffix_context:
            prompt += self.suffix_context

        prompt = dedent(prompt)
        return prompt

    @property
    def max_seed_word_samples(self) -> int:
        r"""Maximum number of seed words to sample across all domain components."""
        return self._max_seed_word_samples

    @property
    def suffix_context(self) -> Optional[str]:
        f"""Optional added suffix context into the generated prompt."""
        return self._suffix_context

    def _sample_seed_words(self, task: AlignmentTask) -> List[str]:
        domain = task.domain

        total_weight = sum(domain.weights)
        component_weights = [weight / total_weight for weight in domain.weights]

        # Arange the list of all *combined* seed words across all components, along with their net
        # sample probability. The sample probability of a seed words is uniform *within* each component.
        # Note: Seed words may appear in distinct domain components.
        seed_words, sample_probs = [], []
        for i, component in enumerate(domain.components):
            seed_words += component.seed_words
            sample_probs += [component_weights[i] / len(component.seed_words)] * len(
                component.seed_words
            )

        sample_size = min(len(seed_words), self.max_seed_word_samples)
        sampled_indices = np.random.choice(
            len(seed_words), size=sample_size, replace=False, p=sample_probs
        )
        sampled_seed_words = [seed_words[i] for i in sampled_indices]

        # Drop duplicates across domains
        sampled_seed_words = list(set(sampled_seed_words))
        return sampled_seed_words
