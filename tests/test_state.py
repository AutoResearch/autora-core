import logging
import pickle
from typing import Optional, Sequence

import pandas as pd
import sklearn.base
import sklearn.dummy
import sklearn.linear_model
from hypothesis import given, settings
from hypothesis import strategies as st

from autora.state import StandardStateDataClass
from autora.variable import Variable, VariableCollection

from .test_variable import variable_strategy, variablecollection_strategy

logger = logging.getLogger(__name__)


@st.composite
def data_length_strategy(draw, max_value=10_000):
    return draw(st.integers(min_value=0, max_value=max_value))


@st.composite
def series_strategy(
    draw, variable: Optional[Variable] = None, n_entries: Optional[int] = None
):
    if variable is None:
        variable = draw(variable_strategy())
    if n_entries is None:
        n_entries = draw(data_length_strategy())

    logger.debug(f"{variable} {n_entries}")

    if variable.allowed_values is not None:
        value_strategy = st.sampled_from(variable.allowed_values)
    elif variable.value_range is not None:
        min_value, max_value = sorted(variable.value_range)
        if isinstance(min_value, int):
            value_strategy = st.integers(min_value=min_value, max_value=max_value)
        elif isinstance(min_value, float):
            value_strategy = st.floats(min_value=min_value, max_value=max_value)
        else:
            raise NotImplementedError(
                f"value_strategy for {variable.value_range} not implemented"
            )
    else:
        value_strategy = st.floats()
    value = draw(st.lists(value_strategy, min_size=n_entries, max_size=n_entries))
    result = pd.Series(value, name=variable.name)
    logger.debug(result)
    return result


@st.composite
def dataframe_strategy(
    draw,
    variables: Optional[Sequence[Variable]] = None,
    n_entries: Optional[int] = None,
):
    if variables is None:
        variables = draw(st.lists(variable_strategy()))
    if n_entries is None:
        n_entries = draw(data_length_strategy())

    d = {}
    for v in variables:
        series = draw(series_strategy(variable=v, n_entries=n_entries))
        d[v.name] = series
    df = pd.DataFrame(d)
    logger.debug(df)
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


@st.composite
def model_strategy(
    draw,
    variable_collection: Optional[VariableCollection] = None,
    n_entries: Optional[int] = None,
    experiment_data: Optional[pd.DataFrame] = None,
    model: Optional[sklearn.base.BaseEstimator] = None,
):
    if variable_collection is None:
        variable_collection = draw(variablecollection_strategy(num_variables=(1, 1, 0)))
        assert isinstance(variable_collection, VariableCollection)
    if n_entries is None:
        n_entries = draw(st.integers(min_value=10, max_value=100))
    if experiment_data is None:
        experiment_data = draw(
            dataframe_strategy(
                variables=(
                    list(variable_collection.independent_variables)
                    + list(variable_collection.dependent_variables)
                ),
                n_entries=n_entries,
            )
        )
    if model is None:
        model = draw(
            st.sampled_from(
                [
                    sklearn.dummy.DummyRegressor,
                    # sklearn.linear_model.LinearRegression,
                    # sklearn.linear_model.Ridge,
                    # sklearn.linear_model.BayesianRidge,
                ]
            )
        )

    X = experiment_data[[v.name for v in variable_collection.independent_variables]]
    y = experiment_data[[v.name for v in variable_collection.dependent_variables]]

    result = model().fit(X, y)
    return result


@given(st.one_of(series_strategy(), dataframe_strategy()))
def test_core_dataframe_serialize_deserialize(o):
    o_loaded = pickle.loads(pickle.dumps(o))
    assert pd.DataFrame.equals(o, o_loaded)


@settings(max_examples=10)
@given(model_strategy())
def test_model_serialize_deserialize(o):
    o_loaded = pickle.loads(pickle.dumps(o))
    assert o == o_loaded


@given(standard_state_dataclass_strategy())
def test_standard_state_dataclass_serialize_deserialize(o: StandardStateDataClass):
    o_loaded = pickle.loads(pickle.dumps(o))
    assert o.variables == o_loaded.variables
    assert pd.DataFrame.equals(o.conditions, o_loaded.conditions)
    assert pd.DataFrame.equals(o.experiment_data, o_loaded.experiment_data)
