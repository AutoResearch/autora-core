import importlib
import logging
import pathlib
from typing import Optional, Union

import dill
import typer
from typing_extensions import Annotated

from autora.state import State

_logger = logging.getLogger(__name__)


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
    verbose: Annotated[bool, typer.Option(help="Turns on info logging level.")] = False,
    debug: Annotated[bool, typer.Option(help="Turns on debug logging level.")] = False,
):
    _configure_logger(debug, verbose)
    starting_state = _load_state(in_path)
    _logger.info(f"Starting State: {starting_state}")
    function = _load_function(fully_qualified_function_name)
    ending_state = function(starting_state)
    _logger.info(f"Ending State: {ending_state}")
    _dump_state(ending_state, out_path)

    return


def _configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def _load_state(path: Optional[pathlib.Path]) -> Union[State, None]:
    if path is not None:
        _logger.debug(f"_load_state: loading from {path=}")
        with open(path, "rb") as f:
            state_ = dill.load(f)
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


def _dump_state(state_: State, path: Optional[pathlib.Path]) -> None:
    if path is not None:
        _logger.debug(f"_dump_state: dumping to {path=}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            dill.dump(state_, f)
    else:
        _logger.debug(f"_dump_state: {path=} so writing to stdout")
        print(dill.dumps(state_))
    return


if __name__ == "__main__":
    typer.run(main)
