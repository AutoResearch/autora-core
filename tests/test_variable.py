import logging

from hypothesis import given
from hypothesis import strategies as st

from .strategies import variable_strategy, variablecollection_strategy
from .test_serializer import serializer_dump_load_strategy

logger = logging.getLogger(__name__)


@given(
    st.one_of(variable_strategy(), variablecollection_strategy()),
    serializer_dump_load_strategy(),
)
def test_variable_serialize_deserialize(o, dump_load):
    o_loaded = dump_load(o)
    assert o_loaded == o
