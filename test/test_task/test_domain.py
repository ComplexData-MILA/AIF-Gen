import pytest

from aif_gen.task import Domain, DomainComponent
from aif_gen.task.seed_words import get_seed_words


def test_init():
    component_a = DomainComponent('Component A', ['a_foo', 'a_bar', 'a_baz'])
    component_b = DomainComponent('Component B', ['b_foo', 'b_bar'])
    component_c = DomainComponent('Component C', ['c_foo', 'c_bar', 'c_baz', 'c_bat'])

    components = [component_a, component_b, component_c]
    weights = [1.5, 0.7, 0.3]  # Need not be normalized

    domain = Domain(components, weights)
    assert domain.components == components
    assert domain.num_components == len(components)
    assert domain.weights == weights


def test_init_no_weights():
    component_a = DomainComponent('Component A', ['a_foo', 'a_bar', 'a_baz'])
    component_b = DomainComponent('Component B', ['b_foo', 'b_bar'])
    component_c = DomainComponent('Component C', ['c_foo', 'c_bar', 'c_baz', 'c_bat'])

    components = [component_a, component_b, component_c]

    domain = Domain(components)
    assert domain.components == components
    assert domain.num_components == len(components)
    assert domain.weights == [1 / len(components)] * len(components)


def test_init_single_component():
    component = DomainComponent('Component A', ['a_foo', 'a_bar', 'a_baz'])
    components = [component]

    domain = Domain(components)
    assert domain.components == components
    assert domain.num_components == len(components)
    assert domain.weights == [1 / len(components)] * len(components)


def test_init_no_components():
    components = []

    with pytest.raises(ValueError):
        _ = Domain(components)


def test_init_weight_component_shape_mismatch():
    component_a = DomainComponent('Component A', ['a_foo', 'a_bar', 'a_baz'])
    component_b = DomainComponent('Component B', ['b_foo', 'b_bar'])
    component_c = DomainComponent('Component C', ['c_foo', 'c_bar', 'c_baz', 'c_bat'])

    components = [component_a, component_b, component_c]
    weights = [0.25, 0.75]

    with pytest.raises(ValueError):
        _ = Domain(components, weights)


def test_init_negative_weight():
    component_a = DomainComponent('Component A', ['a_foo', 'a_bar', 'a_baz'])
    component_b = DomainComponent('Component B', ['b_foo', 'b_bar'])

    components = [component_a, component_b]
    weights = [0.25, -0.75]

    with pytest.raises(ValueError):
        _ = Domain(components, weights)


def test_init_from_dict():
    component_dict = {
        'Component A': {
            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
            'description': 'A Mock Domain Component',
            'weight': 1.5,
        },
        'Component B': {
            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
            'description': 'B Mock Domain Component',
            'weight': 0.7,
        },
        'Component C': {
            'seed_words': ['c_foo', 'c_bar', 'c_baz'],
            'description': 'C Mock Domain Component',
            'weight': 0.3,
        },
    }

    components, weights = [], []
    for component_name, component_args in component_dict.items():
        components.append(
            DomainComponent(
                component_name,
                component_args['seed_words'],
                component_args['description'],
            )
        )
        weights.append(component_args['weight'])

    domain = Domain.from_dict(component_dict)
    assert domain.num_components == len(component_dict)
    for component in domain.components:
        assert component.name in component_dict
        assert component_dict[component.name]['seed_words'] == component.seed_words
        assert component_dict[component.name]['description'] == component.description
    assert domain.weights == weights


def test_init_from_dict_no_weights():
    component_dict = {
        'Component A': {
            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
            'description': 'A Mock Domain Component',
        },
        'Component B': {
            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
            'description': 'B Mock Domain Component',
        },
        'Component C': {
            'seed_words': ['c_foo', 'c_bar', 'c_baz'],
            'description': 'C Mock Domain Component',
        },
    }

    domain = Domain.from_dict(component_dict)
    assert domain.num_components == len(component_dict)
    for component in domain.components:
        assert component.name in component_dict
        assert component_dict[component.name]['seed_words'] == component.seed_words
        assert component_dict[component.name]['description'] == component.description
    assert domain.weights == [1 / len(component_dict)] * len(component_dict)


def test_init_from_dict_multiple_seed_word_alias():
    component_dict = {'education': {'weight': 0.7}, 'technology': {'weight': 0.3}}
    domain = Domain.from_dict(component_dict)
    assert domain.num_components == len(component_dict)
    for component in domain.components:
        assert component.name in component_dict
        assert component.seed_words == get_seed_words(component.name)
    assert domain.weights == [0.7, 0.3]


def test_to_dict_no_weights():
    component_dict = {
        'Component A': {
            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
            'description': 'A Mock Domain Component',
        },
        'Component B': {
            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
            'description': 'B Mock Domain Component',
        },
        'Component C': {
            'seed_words': ['c_foo', 'c_bar', 'c_baz'],
            'description': 'C Mock Domain Component',
        },
    }

    domain = Domain.from_dict(component_dict)

    # Note: We automatically add uniform weights to the domain if they were not specified
    expected_dict = component_dict
    expected_dict['Component A']['weight'] = 1 / 3
    expected_dict['Component B']['weight'] = 1 / 3
    expected_dict['Component C']['weight'] = 1 / 3
    assert expected_dict == domain.to_dict()


def test_to_dict_with_weights():
    component_dict = {
        'Component A': {
            'seed_words': ['a_foo', 'a_bar', 'a_baz'],
            'description': 'A Mock Domain Component',
            'weight': 1.5,
        },
        'Component B': {
            'seed_words': ['b_foo', 'b_bar', 'b_baz'],
            'description': 'B Mock Domain Component',
            'weight': 0.7,
        },
        'Component C': {
            'seed_words': ['c_foo', 'c_bar', 'c_baz'],
            'description': 'C Mock Domain Component',
            'weight': 0.3,
        },
    }

    components, weights = [], []
    for component_name, component_args in component_dict.items():
        components.append(
            DomainComponent(
                component_name,
                component_args['seed_words'],
                component_args['description'],
            )
        )
        weights.append(component_args['weight'])

    domain = Domain.from_dict(component_dict)

    expected_dict = component_dict
    assert expected_dict == domain.to_dict()
