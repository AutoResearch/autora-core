import pickle
from collections import namedtuple
from typing import Any, List

import dill
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from autora.state import StateDataClass


@given(st.sampled_from([StateDataClass()]))
def test_statedict_serialize_deserialize(s):
    s_dumped = pickle.dumps(s)
    s_loaded = pickle.loads(s_dumped)
    assert s_loaded == s


REPLACE = "replace"
EXTEND = "extend"
APPEND = "append"


@st.composite
def type_delta_strategy(draw):
    outer_type = {list: [REPLACE, EXTEND, APPEND]}

    allowable_combinations = {
        bool: [REPLACE],
        int: [REPLACE],
        str: [REPLACE],
        dict: [REPLACE, EXTEND],
        list: [REPLACE, EXTEND, APPEND],
        pd.DataFrame: [REPLACE, EXTEND],
        Any: [REPLACE],
    }

    st.sampled_from([(None, REPLACE), (list, REPLACE), (list, EXTEND), (list, APPEND)])


_FIELD_DEF = namedtuple("_FIELD_DEF", ["type_", "delta"])


@st.composite
def field_strategy(draw):
    outer_type_strategy = st.sampled_from([None, List])
    core_type_strategy = st.sampled_from(
        [
            bool,
            int,
            str,
            # dict, # TODO: add support for this
            # pd.DataFrame, # TODO: add support for this
            # Any  # TODO: add support for this
        ]
    )
    value_strategies = {
        bool: st.booleans,
        int: st.integers,
        str: st.text,
        List: st.lists,
    }
    default_factory_strategies = {List: list}

    delta_strategy = st.sampled_from([REPLACE, EXTEND, APPEND])
    outer_type, inner_type, delta = draw(
        st.tuples(outer_type_strategy, core_type_strategy, delta_strategy)
    )

    if outer_type is None:
        merged_type = inner_type
        field_def = {"type_": merged_type, "delta": delta}
        value_field = draw(st.sampled_from(["value", "default"]))
        value_strategy = value_strategies[inner_type]
        field_def[value_field] = draw(value_strategy())

    elif outer_type is not None:
        merged_type = outer_type[inner_type]
        field_def = {"type_": merged_type, "delta": delta}

        value_field = draw(st.sampled_from(["value", "default", "default_factory"]))
        if value_field == "default_factory":
            field_def[value_field] = default_factory_strategies[outer_type]
        else:
            value_strategy = value_strategies[outer_type](
                value_strategies[inner_type]()
            )
            field_def[value_field] = draw(value_strategy)

    return field_def


@st.composite
def state_object_strategy(draw):
    variable_name_strategy = st.from_regex("\A[_A-Za-z][_A-Za-z0-9]*\Z")

    field_names = draw(st.lists(variable_name_strategy, unique=True))
    field_names_defs = {n: draw(field_strategy()) for n in field_names}

    d = StateDataClass()
    for name, f in field_names_defs.items():
        d = d.add_field(name=name, **f)

    return d


@given(state_object_strategy())
def test_statedict_serialize_deserialize_data(s):
    s

    s_dumped = pickle.dumps(s)
    s_loaded = pickle.loads(s_dumped)
    assert s_loaded == s


@given(st.data())
def test_draw_sequentially(data):
    x = data.draw(state_object_strategy())


@given(state_object_strategy())
def test_statedict_serialize_deserialize_data_dill(s):
    s

    s_dumped = dill.dumps(s)
    s_loaded = dill.loads(s_dumped)
    assert s_loaded == s
