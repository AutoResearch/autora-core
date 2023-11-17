import logging

import pandas as pd
from hypothesis import given, settings

from autora.state import StandardStateDataClass

from .strategies import standard_state_dataclass_strategy
from .test_serializer import SUPPORTED_SERIALIZERS

logger = logging.getLogger(__name__)


@settings(max_examples=1000)
@given(standard_state_dataclass_strategy(), SUPPORTED_SERIALIZERS)
def test_standard_state_dataclass_serialize_deserialize(
    o: StandardStateDataClass, loads_dumps
):
    loads, dumps = loads_dumps
    o_loaded = loads(dumps(o))
    print(o)
    assert o.variables == o_loaded.variables
    assert pd.DataFrame.equals(o.conditions, o_loaded.conditions)
    assert pd.DataFrame.equals(o.experiment_data, o_loaded.experiment_data)
