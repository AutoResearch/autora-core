import importlib
import logging
import pathlib
from collections import namedtuple
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Tuple, Union

from autora.state import State

_logger = logging.getLogger(__name__)


class Supported(str, Enum):
    dill = "dill"
    pickle = "pickle"
    yaml = "yaml"


_SerializerDef = namedtuple(
    "_SerializerDef", ["module", "load", "dump", "dumps", "file_mode"]
)
_serializer_dict: Dict[str, _SerializerDef] = dict(
    pickle=_SerializerDef("pickle", "load", "dump", "dumps", "b"),
    yaml=_SerializerDef(
        "autora.workflow.serializer.yaml_", "load", "dump", "dumps", ""
    ),
    dill=_SerializerDef("dill", "load", "dump", "dumps", "b"),
)


def get_serializer_mode(
    serializer: Supported, interface: Literal["load", "dump", "dumps"]
) -> Tuple[Callable, str]:
    serializer_def = _serializer_dict[serializer]
    module = serializer_def.module
    interface_function_name = getattr(serializer_def, interface)
    _logger.debug(
        f"get_serializer_mode: loading {interface_function_name=} from" f" {module=}"
    )
    module = importlib.import_module(module)
    function = getattr(module, interface_function_name)
    file_mode = serializer_def.file_mode
    return function, file_mode


def load_state(
    path: Optional[pathlib.Path],
    loader: Supported = Supported.dill,
) -> Union[State, None]:
    if path is not None:
        load, file_mode = get_serializer_mode(loader, "load")
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
    dumper: Supported = Supported.dill,
) -> None:
    if path is not None:
        dump, file_mode = get_serializer_mode(dumper, "dump")
        _logger.debug(f"dump_state: dumping to {path=}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, f"w{file_mode}") as f:
            dump(state_, f)
    else:
        dumps, _ = get_serializer_mode(dumper, "dumps")
        _logger.debug(f"dump_state: {path=} so writing to stdout")
        print(dumps(state_))
    return


def load_function(fully_qualified_function_name: str):
    _logger.debug(f"load_function: Loading function {fully_qualified_function_name}")
    module_name, function_name = fully_qualified_function_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    _logger.debug(f"load_function: Loaded function {function} from {module}")
    return function
