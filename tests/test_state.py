import logging
import pickle

import pandas as pd
from hypothesis import given

from autora.state import StandardStateDataClass

from .strategies import standard_state_dataclass_strategy

logger = logging.getLogger(__name__)


@given(standard_state_dataclass_strategy())
def test_standard_state_dataclass_serialize_deserialize(o: StandardStateDataClass):
    o_loaded = pickle.loads(pickle.dumps(o))
    assert o.variables == o_loaded.variables
    assert pd.DataFrame.equals(o.conditions, o_loaded.conditions)
    assert pd.DataFrame.equals(o.experiment_data, o_loaded.experiment_data)
