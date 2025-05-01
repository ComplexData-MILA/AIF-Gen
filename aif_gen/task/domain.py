from typing import Any, Dict, List, Optional

from .domain_component import DomainComponent


class Domain:
    r"""A domain is a combination of DomainComponents along with a weight for each component.

    Note: We do not enforce that the weights be normalized.

    Args:
        components (List[DomainComponent]): List of DomainComponents that constitute the domain.
        weights (Optional[List[float]]): Weights given to each constituent component (uniform if not specified).

    Raises:
        ValueError: If the list of components is empty.
        ValueError: If the number of weights matches the number of components.
        ValueError: If any of the provided weights are negative.
    """

    def __init__(
        self, components: List[DomainComponent], weights: Optional[List[float]] = None
    ) -> None:
        if not len(components):
            raise ValueError(
                'Cannot initialize a Domain with an empty list of DomainComponents'
            )

        if weights is None:
            weights = [1 / len(components)] * len(components)

        if len(weights) != len(components):
            raise ValueError(
                f'Number of components and weights must match, but got {len(components)} components and {len(weights)} weights'
            )
        for i, weight in enumerate(weights):
            if weight < 0:
                raise ValueError(
                    f'Got a negative weight for component: {components[i]}'
                )

        self._components = components
        self._weights = weights

    @classmethod
    def from_dict(cls, domain_dict: Dict[str, Dict[str, Any]]) -> 'Domain':
        r"""Construct an Domain from dictionary represenetation.

        Note:
            Expects each key to denote the 'name' of a DomainComponent, and each value
            to be a dictionary that is parsable by DomainComponent.from_dict().

            Moreover, each value should include a ('weight': weight_value: float) key-value
            pair that encodes the weight for that given DomainComponent.

            If these 'weight' keys are no present, the uniform weight initialization is used.

        Args:
            domain_dict(Dict[str, Dict[str, Any]]): The dictionary that encodes the Domain.

        Returns:
            Domain: The newly constructed Domain.

        Raises:
            ValueError: If the input dictionary is missing any required keys.
        """
        components, component_weights = [], []
        for component_name, component_args in domain_dict.items():
            if component_args is None:
                # Use domain component alias for seed words
                component = DomainComponent.from_dict({'name': component_name})
            else:
                component_args['name'] = component_name
                component = DomainComponent.from_dict(component_args)
                if 'weight' in component_args:
                    component_weights.append(component_args['weight'])

            components.append(component)

        weights = None if not len(component_weights) else component_weights
        return cls(components, weights)

    def to_dict(self) -> Dict[str, Any]:
        r"""Convert the Domain to dictionary represenetation.

        Note: This method is the functional inverse of Domain.from_dict().

        Returns:
            Dict[str, Any]: The dictionary representation of the Domain.
        """
        domain_dict = {}
        for i in range(self.num_components):
            domain_dict[self.components[i].name] = self.components[i].to_dict()
            domain_dict[self.components[i].name]['weight'] = self.weights[i]
        return domain_dict

    def __str__(self) -> str:
        s = f'Domain: ['
        for i in range(self.num_components):
            s += f'({self.components[i]}, weight={self.weights[i]:.2f}), '
        s += ']'
        return s

    @property
    def components(self) -> List[DomainComponent]:
        r"""List[DomainComponent]: The list of components associated with this Domain."""
        return self._components

    @property
    def num_components(self) -> int:
        r"""int: The number of components associated with this Domain."""
        return len(self.components)

    @property
    def weights(self) -> List[float]:
        r"""List[float]: The weights associated with this Domain."""
        return self._weights
