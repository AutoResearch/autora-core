import logging
import pathlib
from typing import Optional

import typer
from typing_extensions import Annotated

from autora.workflow.serializer import Supported as SerializersSupported
from autora.workflow.serializer import dump_state, load_function, load_state

_logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    typer.run(main)
