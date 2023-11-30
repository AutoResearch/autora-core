import logging

from hypothesis import given, settings
from hypothesis import strategies as st

from .test_serializer import serializer_dump_load_strategy
from .test_strategies import variable_strategy, variablecollection_strategy

logger = logging.getLogger(__name__)


@given(
    st.one_of(
        variable_strategy(),
        variablecollection_strategy(),
    ),
    serializer_dump_load_strategy,
)
@settings(deadline=1000)
def test_variable_serialize_deserialize(o, dump_load):
    o_loaded = dump_load(o)
    assert o_loaded == o
