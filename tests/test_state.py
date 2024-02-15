import logging

import pandas as pd
from hypothesis import given
from pandas import DataFrame

from autora.state import StandardState
from autora.variable import Variable, VariableCollection

from .test_serializer import serializer_dump_load_strategy
from .test_strategies import (
    dataframe_strategy,
    standard_state_strategy,
    variable_strategy,
    variablecollection_strategy,
)

logger = logging.getLogger(__name__)


@given(variable_strategy(), serializer_dump_load_strategy)
def test_variable_serialize_deserialize(o: Variable, dump_load):
    o_loaded = dump_load(o)
    assert o == o_loaded


@given(variablecollection_strategy(), serializer_dump_load_strategy)
def test_variablecollection_serialize_deserialize(o: VariableCollection, dump_load):
    o_loaded = dump_load(o)
    assert o == o_loaded


@given(dataframe_strategy(), serializer_dump_load_strategy)
def test_dataframe_serialize_deserialize(o: DataFrame, dump_load):
    o_loaded = dump_load(o)
    o.equals(o_loaded)


@given(standard_state_strategy(), serializer_dump_load_strategy)
def test_state_serialize_deserialize(o: StandardState, dump_load):
    o_loaded = dump_load(o)
    assert o.variables == o_loaded.variables
    assert pd.DataFrame.equals(o.conditions, o_loaded.conditions)
    assert pd.DataFrame.equals(o.experiment_data, o_loaded.experiment_data)
