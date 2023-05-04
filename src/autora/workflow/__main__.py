import logging
import pathlib
import pickle
from typing import Any, Optional

import dill
import typer

from .controller import Controller

_logger = logging.getLogger(__name__)


def main(
    input_path: pathlib.Path = typer.Argument(
        ..., help="Input path, a .dill file with the " "starting state of the manager"
    ),
    output_path: pathlib.Path = typer.Argument(
        ..., help="Output path, a .dill file with the " "ending state of the manager"
    ),
    step_name: Optional[str] = typer.Argument(None, help="Name of step"),
    verbose: bool = typer.Option(False, help="Turns on info logging level."),
    debug: bool = typer.Option(False, help="Turns on debug logging level."),
):
    _logger.debug("initializing")
    _configure_logger(debug, verbose)
    controller_ = _load_manager(input_path)

    if step_name is not None:
        controller_ = _set_next_step_name(controller_, step_name)

    _logger.info("running next step")
    next(controller_)

    _logger.debug(f"last result: {controller_.state.history[-1]}")

    _logger.info("writing out results")
    _dump_manager(controller_, output_path)

    return


def _configure_logger(debug, verbose):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        _logger.debug("using DEBUG logging level")
    if verbose:
        logging.basicConfig(level=logging.INFO)
        _logger.info("using INFO logging level")


def _load_manager(path: pathlib.Path) -> Controller:
    _logger.debug(f"_load_manager: loading from {path=}")
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            controller_ = pickle.load(f)
    elif path.suffix == ".dill":
        with open(path, "rb") as f:
            controller_ = dill.load(f)
    else:
        raise NotImplementedError(f"{path.suffix=} cannot be loaded")
    return controller_


def _dump_manager(controller_: Any, path: pathlib.Path) -> None:
    _logger.debug(f"_dump_manager: dumping to {path=}")

    if path.suffix == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(controller_, f)
    elif path.suffix == ".dill":
        with open(path, "wb") as f:
            dill.dump(controller_, f)
    else:
        raise NotImplementedError(f"{path.suffix=} cannot be dumped")
    return


def _set_next_step_name(controller: Controller, step_name: str):
    _logger.info(f"setting next {step_name=}")
    controller.planner = lambda _: step_name
    return controller


if __name__ == "__main__":
    typer.run(main)
