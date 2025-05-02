from abc import ABC, abstractmethod

from aif_gen.task import AlignmentTask

ETHICAL_GUIDELINES: str = """Ensure that the generated response adheres to ethical practices, avoids biases, and respects the target audience's needs.\n"""


class PromptMapperBase(ABC):
    @abstractmethod
    def generate_prompt(self, task: AlignmentTask) -> str:
        r"""Generate a prompt that, when given to a language model, produces a prompt for a given AlignmentTask.

        Args:
            task (AlignmentTask): The alignment task containing the domain, objective, and preferences.

        Returns:
            str: A structured prompt string for the LLM.
        """


class ResponseMapperBase(ABC):
    @abstractmethod
    def generate_prompt(
        self,
        task: AlignmentTask,
        task_prompt: str,
    ) -> str:
        r"""Generate a prompt that, when given to a language model, produces a (chosen, rejected)
        response pair for the task_prompt and AlignmentTask.

        Args:
            task (AlignmentTask): The alignment task containing the domain, objective, and preferences.
            task_prompt (str): The task prompt to generated responses for.

        Returns:
            str: A structured prompt string for the LLM.
        """
