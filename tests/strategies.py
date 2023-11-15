import logging
import pickle
from typing import Optional, Sequence, Tuple

import pandas as pd
import sklearn.base
import sklearn.dummy
from hypothesis import strategies as st

from autora.state import StandardStateDataClass
from autora.variable import ValueType, Variable, VariableCollection

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
MAX_VARIABLES = 100  # Max 100 variables in total, for speed of testing
MAX_DATA_LENGTH = 1000


@st.composite
def data_length_strategy(draw, max_value=MAX_DATA_LENGTH):
    return draw(st.integers(min_value=0, max_value=max_value))


@st.composite
def variable_strategy(draw, value_strategy=None):
    if value_strategy is None:
        value_strategy = draw(DEFAULT_VALUE_STRATEGY)

    v = Variable(
        name=draw(st.text()),
        variable_label=draw(st.text()),
        units=draw(st.text()),
        type=draw(st.sampled_from(ValueType)),
        is_covariate=draw(st.booleans()),
        value_range=draw(st.one_of(st.none(), value_strategy)),
        allowed_values=draw(
            st.one_of(st.none(), st.lists(value_strategy, unique=True, min_size=0))
        ),
        rescale=draw(st.one_of(st.just(1), value_strategy.filter(lambda v: v != 0))),
    )
    return v


@st.composite
def series_strategy(
    draw,
    variable: Optional[Variable] = None,
    n_entries: Optional[int] = None,
    value_strategy: Optional[st.SearchStrategy] = None,
):
    if variable is None:
        variable = draw(variable_strategy())
    if n_entries is None:
        n_entries = draw(data_length_strategy())
    if value_strategy is None:
        value_strategy = draw(DEFAULT_VALUE_STRATEGY)

    value = draw(st.lists(value_strategy, min_size=n_entries, max_size=n_entries))
    result = pd.Series(value, name=variable.name)
    logger.debug(result)
    return result


@st.composite
def variablecollection_strategy(
    draw,
    max_length=MAX_VARIABLES,
    num_variables: Optional[Tuple[int, int, int]] = None,
    value_strategy: Optional[st.SearchStrategy] = None,
):
    if num_variables is not None:
        num_ivs, num_dvs, n_covariates = num_variables
    else:  # num_variables is None
        num_ivs, num_dvs, n_covariates = draw(
            st.tuples(
                st.integers(min_value=0), st.integers(min_value=0), st.just(0)
            ).filter(lambda n: sum(n) <= max_length)
        )

    n_variables = sum((num_ivs, num_dvs, n_covariates))

    if value_strategy is None:
        value_strategy = draw(DEFAULT_VALUE_STRATEGY)

    all_variables = draw(
        st.lists(
            variable_strategy(value_strategy=value_strategy),
            unique_by=lambda v: v.name,
            min_size=n_variables,
            max_size=n_variables,
        )
    )

    vc = VariableCollection(
        independent_variables=all_variables[0:num_ivs],
        dependent_variables=all_variables[num_ivs : num_ivs + num_dvs],
        covariates=all_variables[num_ivs + num_dvs :],
    )
    return vc


@st.composite
def dataframe_strategy(
    draw,
    variables: Optional[Sequence[Variable]] = None,
    n_entries: Optional[int] = None,
    value_strategy: Optional[st.SearchStrategy] = None,
):
    if variables is None:
        variables = draw(st.lists(variable_strategy()))
    if n_entries is None:
        n_entries = draw(data_length_strategy())
    if value_strategy is None:
        value_strategy = draw(DEFAULT_VALUE_STRATEGY)

    d = {}
    for v in variables:
        series = draw(
            series_strategy(
                variable=v, n_entries=n_entries, value_strategy=value_strategy
            )
        )
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
    value_strategy: Optional[st.SearchStrategy] = None,
    model: Optional[sklearn.base.BaseEstimator] = None,
):
    if variable_collection is None:
        variable_collection = draw(variablecollection_strategy(num_variables=(1, 1, 0)))
        assert isinstance(variable_collection, VariableCollection)
    if n_entries is None:
        n_entries = draw(st.integers(min_value=10, max_value=100))
    if value_strategy is None:
        value_strategy = draw(DEFAULT_VALUE_STRATEGY)()

    if experiment_data is None:
        experiment_data = draw(
            dataframe_strategy(
                variables=(
                    list(variable_collection.independent_variables)
                    + list(variable_collection.dependent_variables)
                ),
                n_entries=n_entries,
                value_strategy=value_strategy,
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


SUPPORTED_SERIALIZERS = st.sampled_from([(pickle.loads, pickle.dumps)])

if __name__ == "__main__":
    print(dataframe_strategy().example())
