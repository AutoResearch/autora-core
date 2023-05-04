"""  The cycle controller for AER. """
from __future__ import annotations

import logging
from typing import Callable, Generic, Mapping, Optional

from .protocol import State

_logger = logging.getLogger(__name__)


class BaseController(Generic[State]):
    """
    Runs an experimentalist, theorist and experiment runner in a loop.

    Once initialized, the `controller` can be started by calling `next(controller)` or using the
        `controller.run` method.

    Attributes:
        state (CycleState or CycleStateHistory): an object which is updated during the cycle and
            is compatible with the `executor_collection`, `planner` and `monitor`.

        planner: a function which takes the `state` as input and returns the name one of the
            `executor_collection` names.

        executor_collection: a mapping between names and functions which take the state as
            input and return a state.

        monitor (Callable): a function which takes the state as input and is called at
            the end of each step.

    """

    def __init__(
        self,
        state: State,
        planner: Callable[[State], str],
        executor_collection: Mapping[str, Callable[[State], State]],
        monitor: Optional[Callable[[State], None]] = None,
    ):
        """
        Args:
            state: a fully instantiated controller state object compatible with the planner,
                executor_collection and monitor
            planner: a function which maps from the state to the next ExecutorName
            executor_collection: a mapping from the ExecutorName to an experiment_runner
                which can operate on the state and return an updated state
            monitor: a function which takes the state object as input
        """

        self.state = state
        self.planner = planner
        self.executor_collection = executor_collection
        self.monitor = monitor

    def run_once(self, step_name: Optional[str] = None):
        """Run one step in the workflow, either by name or using the planner (default)."""

        # Plan
        if step_name is None:
            step_name = self.planner(self.state)

        # Map
        _logger.info(f"getting {step_name=}")
        next_function = self.executor_collection[step_name]
        _logger.info(f"running {next_function=}")
        next_params = self.state.params.get(step_name, {})
        _logger.debug(f"{next_params=}")

        # Execute
        result = next_function(self.state, params=next_params)
        _logger.debug(f"{result=}")

        # Update
        _logger.debug(f"updating state")
        self.state = result

        if self.monitor is not None:
            self.monitor(self.state)
        return self

    def run(self, num_steps: int = 1):
        """Run the next num_steps planned steps in the workflow."""
        for i in range(num_steps):
            self.run_once(self)
        return self

    def __next__(self):
        """Run the next planned step in the workflow."""
        return self.run_once()

    def __iter__(self):
        return self
