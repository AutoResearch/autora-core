import logging

from hypothesis import given
from hypothesis import strategies as st

from .strategies import (
    SUPPORTED_SERIALIZERS,
    variable_strategy,
    variablecollection_strategy,
)

logger = logging.getLogger(__name__)


@given(
    st.one_of(variable_strategy(), variablecollection_strategy()),
    SUPPORTED_SERIALIZERS,
)
def test_dataclass_serialize_deserialize(o, loads_dumps):
    loads, dumps = loads_dumps
    o_loaded = loads(dumps(o))
    assert o_loaded == o
