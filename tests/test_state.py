import logging

import pandas as pd
from hypothesis import HealthCheck, given, settings

from autora.state import StandardStateDataClass

from .test_serializer import serializer_dump_load_strategy
from .test_strategies import standard_state_dataclass_strategy

logger = logging.getLogger(__name__)


@given(standard_state_dataclass_strategy(), serializer_dump_load_strategy)
@settings(suppress_health_check={HealthCheck.too_slow}, deadline=500)
def test_state_serialize_deserialize(o: StandardStateDataClass, dump_load):
    o_loaded = dump_load(o)
    assert o.variables == o_loaded.variables
    assert pd.DataFrame.equals(o.conditions, o_loaded.conditions)
    assert pd.DataFrame.equals(o.experiment_data, o_loaded.experiment_data)
