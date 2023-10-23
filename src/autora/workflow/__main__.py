import importlib
import logging
import pathlib
from collections import namedtuple
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import typer
from typing_extensions import Annotated

from autora.state import State

_logger = logging.getLogger(__name__)


class SerializersSupported(str, Enum):
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


def main(
    fully_qualified_function_name: Annotated[
        str, typer.Argument(help="Function to load")
    ],
    in_path: Annotated[
        Optional[pathlib.Path],
        typer.Option(help="Path to a .dill file with the initial state"),
    ] = None,
    in_loader: Annotated[
        SerializersSupported,
        typer.Option(
            help="deserializer to use to load the data",
        ),
    ] = SerializersSupported.dill,
    out_path: Annotated[
        Optional[pathlib.Path],
        typer.Option(help="Path to output the final state as a .dill file"),
    ] = None,
    out_dumper: Annotated[
        SerializersSupported,
        typer.Option(
            help="serializer to use to save the data",
        ),
    ] = SerializersSupported.dill,
    verbose: Annotated[bool, typer.Option(help="Turns on info logging level.")] = False,
    debug: Annotated[bool, typer.Option(help="Turns on debug logging level.")] = False,
):
    """Run an arbitrary function (on an optional State object) and store the output."""
    _configure_logger(debug, verbose)
    starting_state = load_state(in_path, in_loader)
    _logger.info(f"Starting State: {starting_state}")
    function = load_function(fully_qualified_function_name)
    ending_state = function(starting_state)
    _logger.info(f"Ending State: {ending_state}")
    dump_state(ending_state, out_path, out_dumper)

    return


def _configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def get_serializer_mode(
    serializer: SerializersSupported, interface: Literal["load", "dump", "dumps"]
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
    loader: SerializersSupported = SerializersSupported.dill,
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


def load_function(fully_qualified_function_name: str):
    _logger.debug(f"load_function: Loading function {fully_qualified_function_name}")
    module_name, function_name = fully_qualified_function_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    _logger.debug(f"load_function: Loaded function {function} from {module}")
    return function


def dump_state(
    state_: State,
    path: Optional[pathlib.Path],
    dumper: SerializersSupported = SerializersSupported.dill,
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


if __name__ == "__main__":
    typer.run(main)
