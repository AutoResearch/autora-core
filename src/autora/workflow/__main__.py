import importlib
import logging
import pathlib

import dill
import typer
from typing_extensions import Annotated

from autora.state import State

_logger = logging.getLogger(__name__)


def main(
    fully_qualified_function_name: Annotated[
        str, typer.Argument(help="Function to load")
    ],
    input_path: Annotated[
        pathlib.Path, typer.Argument(help="Path to a .dill file with the initial state")
    ],
    output_path: Annotated[
        pathlib.Path,
        typer.Argument(help="Path to output the final state as a .dill file"),
    ],
    verbose: Annotated[bool, typer.Option(help="Turns on info logging level.")] = False,
    debug: Annotated[bool, typer.Option(help="Turns on debug logging level.")] = False,
):
    _logger.info("Initializing")
    _configure_logger(debug, verbose)

    starting_state = _load_state(input_path)
    _logger.info(f"Starting State: {starting_state}")

    module_name, function_name = fully_qualified_function_name.rsplit(".", 1)

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    ending_state = function(starting_state)

    _logger.info(f"Ending State: {ending_state}")

    _logger.info("Writing out results")
    _dump_state(ending_state, output_path)

    return


def _configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def _load_state(path: pathlib.Path) -> State:
    _logger.debug(f"_load_state: loading from {path=}")
    with open(path, "rb") as f:
        state_ = dill.load(f)
    return state_


def _dump_state(state_: State, path: pathlib.Path) -> None:
    _logger.debug(f"_dump_state: dumping to {path=}")
    with open(path, "wb") as f:
        dill.dump(state_, f)


if __name__ == "__main__":
    typer.run(main)
