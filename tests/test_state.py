import pickle
from typing import Sequence

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from autora.state import StandardStateDataClass
from autora.variable import ValueType, Variable, VariableCollection


@st.composite
def variable_strategy(draw):
    v = Variable(
        name=draw(st.text()),
        type=draw(st.sampled_from(ValueType)),
        value_range=draw(
            st.one_of(
                st.none(),
                st.tuples(st.integers(), st.integers()),
                st.tuples(
                    st.floats(allow_nan=False), st.floats(allow_nan=False)
                ).filter(lambda v: v[0] <= v[1]),
            )
        ),
        allowed_values=draw(
            st.one_of(
                st.none(),
                st.lists(
                    st.one_of(st.booleans(), st.integers(), st.floats(allow_nan=False)),
                    unique=True,
                    min_size=1,
                ),
            )
        ),
        units=draw(st.text()),
        variable_label=draw(st.text()),
        rescale=draw(
            st.one_of(
                st.just(1),
                st.integers(),  # TODO: filter zeros
                st.floats(allow_infinity=False, allow_nan=False),  # TODO: filter zeros
            )
        ),
        is_covariate=draw(st.just(False)),
    )
    return v


@st.composite
def variablecollection_strategy(draw):
    all_variables = draw(
        st.lists(variable_strategy(), unique_by=lambda v: v.name, min_size=0)
    )
    num_variables = len(all_variables)
    num_ivs = draw(st.integers(min_value=0, max_value=num_variables))
    # TODO: Add support for covariates
    num_ivs, num_dvs = num_ivs, num_variables - num_ivs
    vc = VariableCollection(
        independent_variables=all_variables[0:num_ivs],
        dependent_variables=all_variables[num_ivs : num_ivs + num_dvs],
        covariates=[],
    )
    return vc


@st.composite
def dataframe_strategy(draw, variables: Sequence[Variable]):
    n_entries = draw(st.integers(min_value=0))
    d = {}
    for v in variables:
        if v.allowed_values is not None:
            value_strategy = st.sampled_from(v.allowed_values)
        elif v.value_range is not None:
            min_value, max_value = sorted(v.value_range)
            if isinstance(min_value, int):
                value_strategy = st.integers(min_value=min_value, max_value=max_value)
            elif isinstance(min_value, float):
                value_strategy = st.floats(min_value=min_value, max_value=max_value)
        else:
            value_strategy = st.floats()
        d[v.name] = draw(
            st.lists(value_strategy, min_size=n_entries, max_size=n_entries)
        )
    df = pd.DataFrame(d)
    return df


@st.composite
def standard_state_dataclass_strategy(draw):
    variable_collection: VariableCollection = draw(variablecollection_strategy())
    conditions = draw(
        dataframe_strategy(variables=variable_collection.independent_variables)
    )
    experiment_data = draw(
        dataframe_strategy(
            variables=(
                variable_collection.independent_variables
                + variable_collection.dependent_variables
            )
        )
    )
    s = StandardStateDataClass(
        variables=variable_collection,
        conditions=conditions,
        experiment_data=experiment_data,
    )
    return s


@given(st.one_of(variable_strategy(), variablecollection_strategy()))
def test_core_dataclasses_serialize_deserialize(o):
    o_dumped = pickle.dumps(o)
    o_loaded = pickle.loads(o_dumped)
    assert o_loaded == o


@settings(print_blob=True)
@given(standard_state_dataclass_strategy())
def test_standard_state_dataclass_serialize_deserialize(o: StandardStateDataClass):
    o_dumped = pickle.dumps(o)
    o_loaded: StandardStateDataClass = pickle.loads(o_dumped)

    assert o.variables == o_loaded.variables
    assert pd.DataFrame.equals(o.conditions, o_loaded.conditions)
    assert pd.DataFrame.equals(o.experiment_data, o_loaded.experiment_data)
