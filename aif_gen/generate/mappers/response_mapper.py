import random
from textwrap import dedent
from typing import List, Optional, Tuple

from aif_gen.task import AlignmentTask

from .base import ResponseMapperBase

# Personas used by the candidate-sampling pipeline (RLCD-style contrastive
# prompting, arXiv:2307.12950). The single-model substitute for multi-model
# fan-out: an `aligned` prompt encourages the preference, an `anti_aligned`
# prompt encourages violating it (while staying on-topic), and a `neutral`
# prompt omits the preference instruction entirely.
PERSONA_ALIGNED = 'aligned'
PERSONA_ANTI_ALIGNED = 'anti_aligned'
PERSONA_NEUTRAL = 'neutral'
VALID_PERSONAS = (PERSONA_ALIGNED, PERSONA_ANTI_ALIGNED, PERSONA_NEUTRAL)


class ResponseMapper(ResponseMapperBase):
    r"""Generate a prompt that, when given to a language model, produces a winning and losing response to the task_prompt.

    Args:
        suffix_context (Optional[str]=None): Optional suffix text to add at the end of the generated prompt.
    """

    NUM_PREFERENCE_AXES_SAMPLES: int = 3
    PREFERENCE_INCLUSION_PROB_POS: float = 0.8
    PREFERENCE_INCLUSION_PROB_NEG: float = 0.8

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

    # ------------------------------------------------------------------
    # New "Sample-N → Score → Select" pipeline (RLCD + West-of-N + HelpSteer2)
    # ------------------------------------------------------------------
    def generate_candidate_prompt(
        self,
        task: AlignmentTask,
        task_prompt: str,
        persona: str,
    ) -> str:
        r"""Generate a single-response prompt conditioned on a persona.

        This is the RLCD-style contrastive prompting (arXiv:2307.12950): instead
        of asking one model in one call to produce both chosen and rejected,
        we sample N independent responses with opposing instructions. The
        difference in the *prompts* is what produces meaningfully differentiated
        outputs from a single base model.

        Args:
            task: AlignmentTask containing objective, preference, domain.
            task_prompt: The prompt the response should answer.
            persona: One of 'aligned', 'anti_aligned', 'neutral'.

        Returns:
            Prompt string for a single response generation call.
        """
        if persona not in VALID_PERSONAS:
            raise ValueError(
                f'persona must be one of {VALID_PERSONAS}, got {persona!r}'
            )

        if persona == PERSONA_ALIGNED:
            preference_clause = (
                f"You MUST strictly follow this preference in every aspect of your response: '{task.preference}'.\n"
                'Make the preference clearly evident in style, tone, and content.\n'
            )
        elif persona == PERSONA_ANTI_ALIGNED:
            preference_clause = (
                f"You MUST deliberately violate this preference while still answering the prompt and staying on-topic: '{task.preference}'.\n"
                'Do not adopt the preferred style/tone/content; produce something that clearly does not satisfy the preference.\n'
            )
        else:  # neutral
            preference_clause = 'Answer the prompt naturally without considering any particular stylistic preference.\n'

        prompt = f"""\
        Generate a single response to the following prompt: '{task_prompt}'.
        {preference_clause}You don't need to start your response by saying "here is the response" nor to give any meta-explanation. Just provide the response.
        """
        if self.suffix_context:
            prompt += self.suffix_context
        return dedent(prompt)

    @staticmethod
    def default_persona_schedule(
        n_candidates: int, base_temperature: float = 1.0
    ) -> List[Tuple[str, float]]:
        r"""Default (persona, temperature) schedule for N candidate samples.

        Mixes aligned / anti_aligned / neutral with low and high temperatures
        so the pool spans a wide rubric-score range. Deterministic given N.

        Args:
            n_candidates: Number of candidates per prompt.
            base_temperature: Center temperature; temperatures are jittered
                around this value.

        Returns:
            List of (persona, temperature) of length n_candidates.
        """
        if n_candidates < 2:
            raise ValueError(f'n_candidates must be >= 2, got {n_candidates}')

        t_lo = max(0.1, base_temperature - 0.3)
        t_hi = min(2.0, base_temperature + 0.2)
        rotation = [
            (PERSONA_ALIGNED, base_temperature),
            (PERSONA_ANTI_ALIGNED, t_hi),
            (PERSONA_NEUTRAL, base_temperature),
            (PERSONA_ALIGNED, t_lo),
            (PERSONA_ANTI_ALIGNED, base_temperature),
            (PERSONA_NEUTRAL, t_hi),
        ]
        # Ensure we always have at least one aligned and one anti_aligned for
        # well-defined contrast even at very small N.
        schedule = [rotation[i % len(rotation)] for i in range(n_candidates)]
        if not any(p == PERSONA_ALIGNED for p, _ in schedule):
            schedule[0] = (PERSONA_ALIGNED, base_temperature)
        if not any(p == PERSONA_ANTI_ALIGNED for p, _ in schedule):
            schedule[-1] = (PERSONA_ANTI_ALIGNED, t_hi)
        return schedule
