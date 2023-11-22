import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn.dummy
import sklearn.linear_model
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np
from hypothesis.extra import pandas as st_pd

from autora.state import StandardStateDataClass
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


DEFAULT_VALUE_STRATEGY = st.sampled_from(
    [
        st.booleans(),
        st.integers(),
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_subnormal=False),
        st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False),
        st.floats(allow_infinity=True, allow_nan=False, allow_subnormal=False),
        st.floats(allow_infinity=True, allow_nan=False, allow_subnormal=True),
        st.text(),
    ]
)
MAX_VARIABLES = 5  # Max 5 variables in each IVs, DVs, Covariates, for speed of testing
MAX_DATA_LENGTH = 1000

FLOAT_STRATEGIES = st.one_of(
    st.integers(),
    st.floats(min_value=0, max_value=1, allow_nan=False, allow_subnormal=False),
    st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False),
    st.floats(allow_infinity=True, allow_nan=False, allow_subnormal=False),
    st.floats(allow_infinity=True, allow_nan=False, allow_subnormal=True),
)

VALUE_TYPE_VALUE_STRATEGY_MAPPING = {
    ValueType.BOOLEAN: st.booleans(),
    ValueType.INTEGER: st.integers(),
    ValueType.REAL: FLOAT_STRATEGIES,
    ValueType.SIGMOID: FLOAT_STRATEGIES,
    ValueType.PROBABILITY: FLOAT_STRATEGIES,
    ValueType.PROBABILITY_SAMPLE: FLOAT_STRATEGIES,
    ValueType.PROBABILITY_DISTRIBUTION: FLOAT_STRATEGIES,
    ValueType.CLASS: st.text(),
}

AVAILABLE_SKLEARN_MODELS_STRATEGY = st.sampled_from(
    [
        sklearn.dummy.DummyRegressor,
        sklearn.linear_model.LinearRegression,
        sklearn.linear_model.Ridge,
        sklearn.linear_model.BayesianRidge,
    ]
)


@st.composite
def data_length_strategy(draw, max_value=MAX_DATA_LENGTH):
    return draw(st.integers(min_value=0, max_value=max_value))


@st.composite
def variable_strategy(
    draw,
    name=None,
    name_max_length=32,
    variable_label_max_length=256,
    units_max_length=32,
):
    if name is None:
        name = draw(st.text(max_size=name_max_length))
    variable_label = draw(st.text(max_size=variable_label_max_length))
    units = draw(st.text(max_size=units_max_length))
    is_covariate = draw(st.booleans())
    type = draw(st.sampled_from(ValueType))
    dtype = VALUE_TYPE_DTYPE_MAPPING[type]
    value_strategy = VALUE_TYPE_VALUE_STRATEGY_MAPPING[type]

    value_range = draw(
        st.one_of(
            st.none(),
            st.tuples(value_strategy, value_strategy).map(sorted),
        )
    )
    allowed_values = draw(
        st.one_of(st.none(), st.lists(value_strategy, unique=True, min_size=1))
    )
    rescale = draw(st.one_of(st.just(1), value_strategy))

    v = Variable(
        name=name,
        variable_label=variable_label,
        units=units,
        type=type,
        is_covariate=is_covariate,
        value_range=value_range,
        allowed_values=allowed_values,
        rescale=rescale,
        data_type=dtype,
    )
    return v


@st.composite
def variablecollection_strategy(
    draw,
    max_length=MAX_VARIABLES,
    num_variables: Optional[Tuple[int, int, int]] = None,
    name_max_length=32,
    **kwargs,
):
    if num_variables is not None:
        n_ivs, n_dvs, n_covariates = num_variables
    else:  # num_variables is None
        n_ivs, n_dvs, n_covariates = draw(
            st.tuples(
                st.integers(min_value=1, max_value=max_length),
                st.integers(min_value=1, max_value=max_length),
                st.integers(min_value=0, max_value=max_length),
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
        draw(variable_strategy(name=names.pop(), **kwargs)) for _ in range(n_ivs)
    ]
    dependent_variables = [
        draw(variable_strategy(name=names.pop(), **kwargs)) for _ in range(n_dvs)
    ]
    covariates = [
        draw(variable_strategy(name=names.pop(), **kwargs)) for _ in range(n_covariates)
    ]

    vc = VariableCollection(
        independent_variables=independent_variables,
        dependent_variables=dependent_variables,
        covariates=covariates,
    )
    return vc


@st.composite
def dataframe_strategy(
    draw,
    variables: Optional[Sequence[Variable]] = None,
):
    if variables is None:
        variable_collection = draw(variablecollection_strategy())
        variables = (
            variable_collection.independent_variables
            + variable_collection.dependent_variables
            + variable_collection.covariates
        )
    df: pd.DataFrame = draw(
        st_pd.data_frames(
            columns=[st_pd.column(name=v.name, dtype=v.data_type) for v in variables],
        )
    )

    return df


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


@st.composite
def standard_state_dataclass_strategy(draw):
    variable_collection: VariableCollection = draw(
        variablecollection_strategy(
            name_max_length=16, units_max_length=16, variable_label_max_length=32
        )
    )
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
    s = StandardStateDataClass(
        variables=variable_collection,
        conditions=conditions,
        experiment_data=experiment_data,
        models=models,
    )
    return s


@given(variable_strategy())
def test_variable_strategy_creation(o):
    assert o


@given(variablecollection_strategy())
def test_variablecollection_strategy_creation(o):
    assert o


@given(dataframe_strategy())
def test_dataframe_strategy_creation(o):
    assert o is not None


@given(model_strategy())
def test_model_strategy_creation(o):
    assert o


@given(standard_state_dataclass_strategy())
def test_standard_state_dataclass_strategy_creation(o):
    assert o


if __name__ == "__main__":
    print(model_strategy().example())
