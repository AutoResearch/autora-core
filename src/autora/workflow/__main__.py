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


SerializerDef = namedtuple(
    "SerializerDef", ["module", "load", "dump", "dumps", "file_mode"]
)


serializer_dict: Dict[str, SerializerDef] = dict(
    pickle=SerializerDef("pickle", "load", "dump", "dumps", "b"),
    yaml=SerializerDef("autora.workflow.serializer.yaml_", "load", "dump", "dumps", ""),
    dill=SerializerDef("dill", "load", "dump", "dumps", "b"),
)


class _Serializer(str, Enum):
    dill = "dill"
    pickle = "pickle"
    yaml = "yaml"


def main(
    fully_qualified_function_name: Annotated[
        str, typer.Argument(help="Function to load")
    ],
    in_path: Annotated[
        Optional[pathlib.Path],
        typer.Option(help="Path to a .dill file with the initial state"),
    ] = None,
    out_path: Annotated[
        Optional[pathlib.Path],
        typer.Option(help="Path to output the final state as a .dill file"),
    ] = None,
    loader: Annotated[
        _Serializer,
        typer.Option(
            help="deserializer to use to load the data",
        ),
    ] = _Serializer.dill,
    dumper: Annotated[
        _Serializer,
        typer.Option(
            help="serializer to use to save the data",
        ),
    ] = _Serializer.dill,
    verbose: Annotated[bool, typer.Option(help="Turns on info logging level.")] = False,
    debug: Annotated[bool, typer.Option(help="Turns on debug logging level.")] = False,
):
    _configure_logger(debug, verbose)
    starting_state = _load_state(in_path, loader)
    _logger.info(f"Starting State: {starting_state}")
    function = _load_function(fully_qualified_function_name)
    ending_state = function(starting_state)
    _logger.info(f"Ending State: {ending_state}")
    _dump_state(ending_state, out_path, dumper)

    return


def _configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def _get_serializer_mode(
    serializer: _Serializer, interface: Literal["load", "dump", "dumps"]
) -> Tuple[Callable, str]:
    serializer_def = serializer_dict[serializer]
    module = serializer_def.module
    interface_function_name = getattr(serializer_def, interface)
    _logger.debug(
        f"_get_serializer: loading {interface_function_name=} from" f" {module=}"
    )
    module = importlib.import_module(module)
    function = getattr(module, interface_function_name)
    file_mode = serializer_def.file_mode
    return function, file_mode


def _load_state(
    path: Optional[pathlib.Path], loader: _Serializer = _Serializer.dill
) -> Union[State, None]:
    if path is not None:
        load, file_mode = _get_serializer_mode(loader, "load")
        _logger.debug(f"_load_state: loading from {path=}")
        with open(path, f"r{file_mode}") as f:
            state_ = load(f)
    else:
        _logger.debug(f"_load_state: {path=} -> returning None")
        state_ = None
    return state_


def _load_function(fully_qualified_function_name: str):
    _logger.debug(f"_load_function: Loading function {fully_qualified_function_name}")
    module_name, function_name = fully_qualified_function_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    _logger.debug(f"_load_function: Loaded function {function} from {module}")
    return function


def _dump_state(
    state_: State,
    path: Optional[pathlib.Path],
    dumper: _Serializer = _Serializer.dill,
) -> None:
    if path is not None:
        dump, file_mode = _get_serializer_mode(dumper, "dump")
        _logger.debug(f"_dump_state: dumping to {path=}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, f"w{file_mode}") as f:
            dump(state_, f)
    else:
        dumps, _ = _get_serializer_mode(dumper, "dumps")
        _logger.debug(f"_dump_state: {path=} so writing to stdout")
        print(dumps(state_))
    return


if __name__ == "__main__":
    typer.run(main)
