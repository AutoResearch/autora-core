import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import sklearn.dummy
import sklearn.linear_model
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np
from hypothesis.extra import pandas as st_pd

from autora.state import StandardState
from autora.variable import ValueType, Variable, VariableCollection

VALUE_TYPE_DTYPE_MAPPING = {
    ValueType.BOOLEAN: bool,
    ValueType.INTEGER: int,
    ValueType.REAL: float,
    ValueType.SIGMOID: float,
    ValueType.PROBABILITY: float,
    ValueType.PROBABILITY_SAMPLE: float,
    ValueType.PROBABILITY_DISTRIBUTION: float,
    ValueType.CLASS: str,
}

logger = logging.getLogger(__name__)

AVAILABLE_SKLEARN_MODELS_STRATEGY = st.sampled_from(
    [
        sklearn.dummy.DummyRegressor,
        sklearn.linear_model.LinearRegression,
        sklearn.linear_model.Ridge,
        sklearn.linear_model.BayesianRidge,
    ]
)


@st.composite
def _name_label_units_strategy(draw, name=None, label=None, units=None, covariate=None):
    if name is None:
        name = draw(st.text(min_size=1))
    if label is None:
        label = draw(st.text(min_size=0))
    if units is None:
        units = draw(st.text(min_size=0))
    if covariate is None:
        covariate = draw(st.booleans())
    return name, label, units, covariate


@st.composite
def variable_boolean_strategy(draw, name=None, label=None, units=None, covariate=None):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.BOOLEAN
    allowed_values = [True, False]
    value_range = None
    rescale = 1
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


@st.composite
def variable_integer_strategy(draw, name=None, label=None, units=None, covariate=None):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.INTEGER

    value_range = draw(
        st.one_of(
            st.none(),
            st.tuples(st.integers(), st.integers())
            .filter(lambda x: x[0] != x[1])
            .map(sorted),
        )
    )
    if value_range is None:
        allowed_values = draw(st.one_of(st.none(), st.sets(st.integers(), min_size=1)))
    else:
        allowed_values = None

    rescale = draw(
        st.one_of(
            st.just(1),
            st.integers(),
            st.floats(allow_infinity=False, allow_subnormal=False, allow_nan=False),
        )
    )
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


@st.composite
def variable_real_strategy(draw, name=None, label=None, units=None, covariate=None):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.REAL
    range_strategy = st.floats(allow_nan=False, allow_subnormal=False)
    value_range = draw(
        st.one_of(
            st.none(),
            st.tuples(range_strategy, range_strategy)
            .filter(lambda x: x[0] != x[1])
            .map(sorted),
        )
    )

    if value_range is None:
        allowed_values = draw(st.one_of(st.none(), st.sets(range_strategy, min_size=1)))
    else:
        allowed_values = None
    rescale = draw(st.one_of(st.just(1), range_strategy))
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


@st.composite
def variable_probability_strategy(
    draw, name=None, label=None, units=None, covariate=None
):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.PROBABILITY
    value_range = (0, 1)
    allowed_values = None
    rescale = 1
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


@st.composite
def variable_probability_sample_strategy(
    draw, name=None, label=None, units=None, covariate=None
):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.PROBABILITY_SAMPLE
    value_range = (0, 1)
    allowed_values = None
    rescale = 1
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


@st.composite
def variable_probability_distribution_strategy(
    draw, name=None, label=None, units=None, covariate=None
):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.PROBABILITY_DISTRIBUTION
    value_range = (0, 1)
    allowed_values = None
    rescale = 1
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


@st.composite
def variable_sigmoid_strategy(draw, name=None, label=None, units=None, covariate=None):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.SIGMOID
    value_range = (-np.inf, +np.inf)
    allowed_values = None
    rescale = 1
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


@st.composite
def variable_class_strategy(draw, name=None, label=None, units=None, covariate=None):
    name, label, units, covariate = draw(
        _name_label_units_strategy(
            name=name, label=label, units=units, covariate=covariate
        )
    )
    value_type = ValueType.CLASS
    value_range = None
    rescale = 1
    allowed_values = draw(st.lists(st.text(min_size=1, max_size=16), unique=True))
    return Variable(
        name=name,
        variable_label=label,
        units=units,
        type=value_type,
        is_covariate=covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
    )


VARIABLE_STRATEGIES = (
    variable_boolean_strategy,
    variable_integer_strategy,
    variable_probability_strategy,
    variable_probability_sample_strategy,
    variable_probability_distribution_strategy,
    variable_sigmoid_strategy,
    variable_real_strategy,
    variable_class_strategy,
)


@st.composite
def variable_strategy(
    draw, elements=VARIABLE_STRATEGIES, value_type: Optional[ValueType] = None, **kwargs
):
    if value_type is None:
        strategy = draw(st.sampled_from(elements))

    else:
        strategy = {
            ValueType.BOOLEAN: variable_boolean_strategy,
            ValueType.INTEGER: variable_integer_strategy,
            ValueType.REAL: variable_real_strategy,
            ValueType.SIGMOID: variable_sigmoid_strategy,
            ValueType.PROBABILITY: variable_probability_strategy,
            ValueType.PROBABILITY_SAMPLE: variable_probability_sample_strategy,
            ValueType.PROBABILITY_DISTRIBUTION: variable_probability_distribution_strategy,
            ValueType.CLASS: variable_class_strategy,
        }[value_type]
    return draw(strategy(**kwargs))


@given(variable_strategy())
def test_variable_strategy_creation(o):
    assert o


@st.composite
def variablecollection_strategy(
    draw,
    elements=VARIABLE_STRATEGIES,
    value_type: Optional[ValueType] = None,
    max_ivs=5,
    max_dvs=1,
    max_covariates=2,
    name_max_length=32,
    **kwargs,
):
    n_ivs, n_dvs, n_covariates = draw(
        st.tuples(
            st.integers(min_value=1, max_value=max_ivs),
            st.integers(min_value=1, max_value=max_dvs),
            st.integers(min_value=0, max_value=max_covariates),
        )
    )

    n_variables = n_ivs + n_dvs + n_covariates

    names = draw(
        st.lists(
            st.text(min_size=1, max_size=name_max_length),
            unique=True,
            min_size=n_variables,
            max_size=n_variables,
        )
    )
    independent_variables = [
        draw(
            variable_strategy(
                name=names.pop(), value_type=value_type, elements=elements, **kwargs
            )
        )
        for _ in range(n_ivs)
    ]
    dependent_variables = [
        draw(
            variable_strategy(
                name=names.pop(), value_type=value_type, elements=elements, **kwargs
            )
        )
        for _ in range(n_dvs)
    ]
    covariates = [
        draw(
            variable_strategy(
                name=names.pop(), value_type=value_type, elements=elements, **kwargs
            )
        )
        for _ in range(n_covariates)
    ]

    vc = VariableCollection(
        independent_variables=independent_variables,
        dependent_variables=dependent_variables,
        covariates=covariates,
    )
    return vc


@given(variablecollection_strategy())
def test_variablecollection_strategy_creation(o):
    assert o


@st.composite
def dataframe_strategy(
    draw,
    variables: Optional[Sequence[Variable]] = None,
    value_type: Optional[ValueType] = None,
):
    if variables is None:
        variable_collection = draw(variablecollection_strategy(value_type=value_type))
        variables = (
            variable_collection.independent_variables
            + variable_collection.dependent_variables
            + variable_collection.covariates
        )

    df: pd.DataFrame = draw(
        st_pd.data_frames(
            columns=[
                st_pd.column(name=v.name, dtype=VALUE_TYPE_DTYPE_MAPPING[v.type])
                for v in variables
            ],
        )
    )

    return df


@given(dataframe_strategy())
def test_dataframe_strategy_creation(o):
    assert o is not None


@st.composite
def model_strategy(draw, models=AVAILABLE_SKLEARN_MODELS_STRATEGY):
    model = draw(models)

    n_x = draw(st.integers(min_value=1, max_value=5))
    n_y = draw(st.integers(min_value=1, max_value=1))
    n_measurements = draw(st.integers(min_value=5, max_value=100))

    elements = st_np.from_dtype(
        np.dtype(float),
        allow_infinity=False,
        allow_subnormal=False,
        allow_nan=False,
        # Include some reasonable extreme values. Values near the upper limit of the float
        # ~10**308, and very small values broke the fitting
        min_value=-1e5,
        max_value=1e5,
        min_magnitude=1e-3,
    )
    X = draw(st_np.arrays(float, shape=(n_measurements, n_x), elements=elements))
    y = draw(st_np.arrays(float, shape=(n_measurements, n_y), elements=elements))

    result = model().fit(X, y.ravel())
    return result


@given(model_strategy())
def test_model_strategy_creation(o):
    assert o


@st.composite
def standard_state_strategy(draw):
    variable_collection: VariableCollection = draw(variablecollection_strategy())
    conditions = draw(
        dataframe_strategy(variables=variable_collection.independent_variables)
    )
    experiment_data = draw(
        dataframe_strategy(
            variables=(
                variable_collection.independent_variables
                + variable_collection.dependent_variables
                + variable_collection.covariates
            )
        )
    )
    models = draw(st.lists(model_strategy(), min_size=0, max_size=5))
    s = StandardState(
        variables=variable_collection,
        conditions=conditions,
        experiment_data=experiment_data,
        models=models,
    )
    return s


@settings(suppress_health_check={HealthCheck.too_slow})
@given(standard_state_strategy())
def test_standard_state_strategy_creation(o):
    assert o


if __name__ == "__main__":
    print(model_strategy().example())
