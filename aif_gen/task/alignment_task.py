from __future__ import annotations

from typing import Any, Dict

from pydantic.dataclasses import dataclass

from .domain import Domain


@dataclass(slots=True, frozen=True)
class AlignmentTask:
    r"""Encapsulates the specification of an alignment problem.

    Args:
        domain (Domain): Domain of the alignment task.
        objective (str): Description of the alignment objective.
        preference (str): Description of the objective preference.
    """

    domain: Domain
    objective: str
    preference: str

    @classmethod
    def from_dict(cls, task_dict: Dict[str, Any]) -> AlignmentTask:
        domain = Domain.from_dict(task_dict['domain'])
        objective = task_dict['objective']
        preference = task_dict['preference']
        return cls(domain, objective, preference)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain': self.domain.to_dict(),
            'objective': self.objective,
            'preference': self.preference,
        }
