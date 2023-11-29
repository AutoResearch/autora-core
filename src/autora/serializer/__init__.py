import importlib
import logging
import pathlib
from collections import namedtuple
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Tuple, Union

from autora.state import State

_logger = logging.getLogger(__name__)


class SerializersSupported(str, Enum):
    """Listing of allowed serializers."""

    pickle = "pickle"
    dill = "dill"
    yaml = "yaml"


# Dictionary of details about each serializer
_SERIALIZER_INFO_ENTRY = namedtuple(
    "_SERIALIZER_INFO_ENTRY", ["module_path", "file_mode"]
)
_SERIALIZER_INFO: Dict[SerializersSupported, _SERIALIZER_INFO_ENTRY] = {
    SerializersSupported.pickle: _SERIALIZER_INFO_ENTRY("pickle", "b"),
    SerializersSupported.dill: _SERIALIZER_INFO_ENTRY("dill", "b"),
    SerializersSupported.yaml: _SERIALIZER_INFO_ENTRY("autora.serializer.yaml_", ""),
}

# Import those serializers which are actually importable
_AVAILABLE_SERIALIZER_INFO = dict()
_LOADED_SERIALIZER_DEF = namedtuple(
    "_LOADED_SERIALIZER_DEF", ["name", "module", "file_mode"]
)
for serializer_enum in SerializersSupported:
    serializer_info = _SERIALIZER_INFO[serializer_enum]
    try:
        module = importlib.import_module(serializer_info.module_path)
    except ImportError:
        _logger.info(f"serializer {serializer_info.module_path} not available")
        continue
    _AVAILABLE_SERIALIZER_INFO[serializer_enum] = _LOADED_SERIALIZER_DEF(
        serializer_info.module_path, module, serializer_info.file_mode
    )

default_serializer = SerializersSupported.pickle


def _get_serializer_mode(
    serializer: SerializersSupported,
    interface: Literal["load", "dump", "loads", "dumps"],
) -> Tuple[Callable, str]:
    serializer_def = _AVAILABLE_SERIALIZER_INFO[serializer]
    module = serializer_def.module
    function = getattr(module, interface)
    file_mode = serializer_def.file_mode
    return function, file_mode


def load_state(
    path: Optional[pathlib.Path],
    loader: SerializersSupported = default_serializer,
) -> Union[State, None]:
    """Load a State object from a path."""
    if path is not None:
        load, file_mode = _get_serializer_mode(loader, "load")
        _logger.debug(f"load_state: loading from {path=}")
        with open(path, f"r{file_mode}") as f:
            state_ = load(f)
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
    if path is not None:
        dump, file_mode = _get_serializer_mode(dumper, "dump")
        _logger.debug(f"dump_state: dumping to {path=}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, f"w{file_mode}") as f:
            dump(state_, f)
    else:
        dumps, _ = _get_serializer_mode(dumper, "dumps")
        _logger.debug(f"dump_state: {path=} so writing to stdout")
        print(dumps(state_))
    return
