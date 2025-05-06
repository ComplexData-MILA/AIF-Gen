from __future__ import annotations

from typing import Any, Dict, List

from pydantic import NonNegativeFloat
from pydantic.dataclasses import dataclass

from .domain_component import DomainComponent


@dataclass(slots=True)
class Domain:
    r"""A domain is a combination of DomainComponents along with a weight for each component.

    Note: We do not enforce that the weights be normalized.

    Args:
        components (List[DomainComponent]): List of DomainComponents that constitute the domain.
        weights (Optional[List[NonNegativeFloat]]): Weights given to each constituent component (uniform if not specified).

    Raises:
        ValueError: If the list of components is empty.
        ValueError: If the number of weights matches the number of components.
        ValueError: If any of the provided weights are negative.
    """

    components: List[DomainComponent]
    weights: List[NonNegativeFloat] | None = None

    def __post_init__(self) -> None:
        if not len(self.components):
            raise ValueError('Domain must have a non-empty list of DomainComponents')
        if self.weights is None:
            self.weights = [1 / self.num_components] * self.num_components
        if len(self.weights) != self.num_components:
            raise ValueError(
                f'Mismatch: {self.num_components} components, {len(self.weights)} weights'
            )

    @classmethod
    def from_dict(cls, domain_dict: Dict[str, Dict[str, Any]]) -> Domain:
        components, component_weights = [], []
        for component_name, component_args in domain_dict.items():
            component_args['name'] = component_name
            components.append(DomainComponent.from_dict(component_args))
            if 'weight' in component_args:
                component_weights.append(component_args['weight'])
        weights = None if not len(component_weights) else component_weights
        return cls(components, weights)

    def to_dict(self) -> Dict[str, Any]:
        assert self.weights is not None
        domain_dict = {}
        for i, component in enumerate(self.components):
            domain_dict[component.name] = component.to_dict()
            domain_dict[component.name]['weight'] = self.weights[i]
        return domain_dict

    @property
    def num_components(self) -> int:
        r"""int: The number of components associated with this Domain."""
        return len(self.components)
