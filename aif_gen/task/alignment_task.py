from typing import Any, Dict

from .domain import Domain


class AlignmentTask:
    r"""Encapsulates the specification of an alignment problem.

    Args:
        domain (Domain): Domain of the alignment task.
        objective (str): Description of the alignment objective.
        preference (str): Description of the objective preference.
    """

    def __init__(self, domain: Domain, objective: str, preference: str) -> None:
        self._domain = domain
        self._objective = objective
        self._preference = preference

    @classmethod
    def from_dict(cls, task_dict: Dict[str, Any]) -> 'AlignmentTask':
        r"""Construct an AlignmentTask from dictionary representation.

        Note:
            Expects 'domain', 'objective', and 'preference' keys to be present in the dictionary.
            Moreover, expects that the 'domain' value is parasable by Domain.from_dict().

        Args:
            task_dict (Dict[str, Any]): The dictionary that encodes the AlignmentTask.

        Returns:
            AlignmentTask: The newly constructed alignmentTask

        Raises:
            ValueError: If the input dictionary is missing any required keys.
        """
        expected_keys = {'domain', 'objective', 'preference'}
        missing_keys = expected_keys - set(task_dict)
        if len(missing_keys):
            raise ValueError(f'Missing required keys: {missing_keys}')

        domain = Domain.from_dict(task_dict['domain'])
        objective = task_dict['objective']
        preference = task_dict['preference']
        return cls(domain, objective, preference)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the AlignmentTask to dictionary represenetation.

        Note: This method is the functional inverse of AlignmentTask.from_dict().

        Returns:
            Dict[str, Any]: The dictionary representation of the alignmentTask.
        """
        return {
            'domain': self.domain.to_dict(),
            'objective': self.objective,
            'preference': self.preference,
        }

    def __str__(self) -> str:
        return f'AlignmentTask({self.domain}, Objective: {self.objective}, Preference: {self.preference})'

    @property
    def domain(self) -> Domain:
        r"""Domain: The domain in the current AlignmentTask."""
        return self._domain

    @property
    def objective(self) -> str:
        r"""str: The objective in the current AlignmentTask."""
        return self._objective

    @property
    def preference(self) -> str:
        r"""str: The preference in the current AlignmentTask."""
        return self._preference
