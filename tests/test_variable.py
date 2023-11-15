import logging
import pickle
from typing import Optional, Tuple

from hypothesis import given
from hypothesis import strategies as st

from autora.variable import ValueType, Variable, VariableCollection

MAX_VARIABLES = 100  # Max 100 variables in total, for speed of testing
SUPPORTED_SERIALIZERS = st.sampled_from([(pickle.loads, pickle.dumps)])

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


@given(
    st.one_of(variable_strategy(), variablecollection_strategy()),
    SUPPORTED_SERIALIZERS,
)
def test_dataclass_serialize_deserialize(o, loads_dumps):
    loads, dumps = loads_dumps
    o_loaded = loads(dumps(o))
    assert o_loaded == o
