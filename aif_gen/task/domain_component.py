from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from pydantic.dataclasses import dataclass

from .seed_words import get_seed_words


@dataclass(slots=True)
class DomainComponent:
    r"""A domain component is an alias for a set of 'seed words' describing a specific body of knowledge.

    Args:
        name (str): The name that describes the content of the domain component.
        seed_words (List[str]): The list of seed words that describe the domain component.
        description (Optional[str]): An optional description of the domain component.

    Raises:
        ValueError: If no seed words are provided.
    """

    name: str
    seed_words: List[str]
    description: str | None = None

    def __post_init__(self) -> None:
        if not len(self.seed_words):
            raise ValueError('DomainComponent seed words list cannot be empty')

    @classmethod
    def from_dict(cls, comp_dict: Dict[str, Any]) -> DomainComponent:
        # If seed words not provided, use the component name itself
        comp_dict['seed_words'] = comp_dict.get('seed_words', comp_dict['name'])
        if isinstance(comp_dict['seed_words'], str):
            comp_dict['seed_words'] = get_seed_words(comp_dict['seed_words'])
        return cls(**comp_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
