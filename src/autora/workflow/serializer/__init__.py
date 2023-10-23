import importlib
from collections import namedtuple
from enum import Enum
from typing import Callable, Dict, Literal, Tuple

from autora.workflow.__main__ import _logger


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
        f"_get_serializer: loading {interface_function_name=} from" f" {module=}"
    )
    module = importlib.import_module(module)
    function = getattr(module, interface_function_name)
    file_mode = serializer_def.file_mode
    return function, file_mode
