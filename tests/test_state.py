import logging
import pickle

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from autora.state import StandardStateDataClass

from .strategies import (
    dataframe_strategy,
    model_strategy,
    series_strategy,
    standard_state_dataclass_strategy,
)

logger = logging.getLogger(__name__)


@given(st.one_of(series_strategy(), dataframe_strategy()))
def test_core_dataframe_serialize_deserialize(o):
    o_loaded = pickle.loads(pickle.dumps(o))
    assert pd.DataFrame.equals(o, o_loaded)


@settings(max_examples=10)
@given(
    model_strategy(
        value_strategy=st.sampled_from(
            [
                st.integers(),
                st.floats(
                    min_value=0, max_value=1, allow_nan=False, allow_subnormal=False
                ),
                st.floats(allow_infinity=False, allow_nan=False, allow_subnormal=False),
            ]
        )
    )
)
def test_model_serialize_deserialize(o):
    o_loaded = pickle.loads(pickle.dumps(o))
    assert o == o_loaded


@given(standard_state_dataclass_strategy())
def test_standard_state_dataclass_serialize_deserialize(o: StandardStateDataClass):
    o_loaded = pickle.loads(pickle.dumps(o))
    assert o.variables == o_loaded.variables
    assert pd.DataFrame.equals(o.conditions, o_loaded.conditions)
    assert pd.DataFrame.equals(o.experiment_data, o_loaded.experiment_data)
