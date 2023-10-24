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


_SerializerDef = namedtuple(
    "_SerializerDef", ["module", "load", "dump", "dumps", "file_mode"]
)
_serializer_dict: Dict[SerializersSupported, _SerializerDef] = {
    SerializersSupported.pickle: _SerializerDef("pickle", "load", "dump", "dumps", "b"),
    SerializersSupported.yaml: _SerializerDef(
        "autora.serializer._yaml", "load", "dump", "dumps", ""
    ),
    SerializersSupported.dill: _SerializerDef("dill", "load", "dump", "dumps", "b"),
}

default_serializer = SerializersSupported.pickle


def _get_serializer_mode(
    serializer: SerializersSupported, interface: Literal["load", "dump", "dumps"]
) -> Tuple[Callable, str]:
    serializer_def = _serializer_dict[serializer]
    module = serializer_def.module
    interface_function_name = getattr(serializer_def, interface)
    _logger.debug(
        f"_get_serializer_mode: loading {interface_function_name=} from" f" {module=}"
    )
    module = importlib.import_module(module)
    function = getattr(module, interface_function_name)
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
