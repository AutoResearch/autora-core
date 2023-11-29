"""A submodule which handles importing supported serializers."""

import importlib
import logging
import pathlib
from collections import namedtuple
from enum import Enum
from typing import Dict, Optional, Union

from autora.state import State

_logger = logging.getLogger(__name__)

# Developer notes:
# Add a new serializer by:
# - including its name in the SerializersSupported Enum. This {name} will be available on the
#   command line: `python -m autora.workflow --serializer {name}`
# - Adding the basic data about it to the _SERIALIZER_INFO dictionary. This should include the
#   fully qualified name under which it can be imported, and which file mode it expects:
#     "" for regular [i.e. files will be opened as `open(filename, "w")` for writing]
#     "b" for binary [i.e. files will be opened as `open(filename, "wb")` for writing]

_SERIALIZER_INFO_ENTRY = namedtuple(
    "_SERIALIZER_INFO_ENTRY", ["fully_qualified_module_name", "file_mode"]
)
LOADED_SERIALIZER = namedtuple("LOADED_SERIALIZER", ["module", "file_mode"])


class SerializersSupported(str, Enum):
    """Listing of allowed serializers."""

    pickle = "pickle"
    dill = "dill"
    yaml = "yaml"


# Dictionary of details about each serializer
_SERIALIZER_INFO: Dict[SerializersSupported, _SERIALIZER_INFO_ENTRY] = {
    SerializersSupported.pickle: _SERIALIZER_INFO_ENTRY("pickle", "b"),
    SerializersSupported.dill: _SERIALIZER_INFO_ENTRY("dill", "b"),
    SerializersSupported.yaml: _SERIALIZER_INFO_ENTRY("autora.serializer.yaml_", ""),
}

# Set the default serializer for the package
default_serializer = SerializersSupported.pickle

# A dictionary to handle lazy loading of the serializers
_LOADED_SERIALIZERS: Dict[SerializersSupported, LOADED_SERIALIZER] = dict()


def load_serializer(serializer: SerializersSupported) -> LOADED_SERIALIZER:
    """Load"""

    try:
        serializer_def = _LOADED_SERIALIZERS[serializer]

    except KeyError:
        serializer_info = _SERIALIZER_INFO[serializer]
        module_ = importlib.import_module(serializer_info.fully_qualified_module_name)
        serializer_def = LOADED_SERIALIZER(module_, serializer_info.file_mode)
        _LOADED_SERIALIZERS[serializer] = serializer_def

    return serializer_def


def load_state(
    path: Optional[pathlib.Path],
    loader: SerializersSupported = default_serializer,
) -> Union[State, None]:
    """Load a State object from a path."""
    serializer = load_serializer(loader)
    if path is not None:
        _logger.debug(f"load_state: loading from {path=}")
        with open(path, f"r{serializer.file_mode}") as f:
            state_ = serializer.module.load(f)
    else:
        _logger.debug(f"load_state: {path=} -> returning None")
        state_ = None
    return state_


def dump_state(
    state_: State,
    path: Optional[pathlib.Path],
    dumper: SerializersSupported = default_serializer,
) -> None:
    """Write a State object to a path."""
    serializer = load_serializer(dumper)
    if path is not None:
        _logger.debug(f"dump_state: dumping to {path=}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, f"w{serializer.file_mode}") as f:
            serializer.module.dump(state_, f)
    else:
        _logger.debug(f"dump_state: {path=} so writing to stdout")
        print(serializer.module.dumps(state_))
    return
