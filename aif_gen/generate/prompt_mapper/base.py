from abc import ABC, abstractmethod
from typing import Any

from aif_gen.task import AlignmentTask


class PromptMapperBase(ABC):
    ETHICAL_GUIDELINES: str = """Ensure that the generated response adheres to ethical practices, avoids biases, and respects the target audience's needs.\n"""

    @abstractmethod
    def generate_prompt(self, task: AlignmentTask, *args: Any, **kwargs: Any) -> str:
        r"""Generate a prompt that, when given to a language model, produces a prompt for a given AlignmentTask.

        Args:
            task (AlignmentTask): The alignment task containing the domain, objective, and preferences.
            args (Any): Optional positional arguments.
            kwargs (Any): Optional keyword arguments.

        Returns:
            str: A structured prompt string for the LLM.
        """

    def __str__(self) -> str:
        r"""Returns the type of PromptMapper."""
        return self.__class__.__name__
