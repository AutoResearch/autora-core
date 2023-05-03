"""
Objects for handling input and outputs from experimentalists, experiment runners and theorists.
"""
import collections
import copy
import logging
import pprint
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from autora.experimentalist.pipeline import Pipeline
from sklearn.base import BaseEstimator

from .protocol import Executor, SupportsControllerState
from .state import resolve_state_params

_logger = logging.getLogger(__name__)


class ChainedFunctionMapping(collections.UserDict):
    """
    Mapping name -> chains of functions applied like decorators.

    Examples:
        We can chain three simple functions to show basic usage:
        >>> f0_ = lambda: 0
        >>> f1_ = lambda _: lambda: 1
        >>> f2_ = lambda _: lambda: 2

        >>> c = ChainedFunctionMapping(
        ...     a=[f0_],          # -> f0_
        ...     b=[f1_, f0_],     # -> f1_(f0_)
        ...     c=[f2_, f1_, f0_] # -> f2_(f1_(f0_))
        ... )

        The "a" chain is the same as `f0_`
        >>> c["a"]()
        0

        The "b" chain is the same as `f1_(f0_)`
        >>> c["b"]()
        1

        The "c" chain is the same as `f2_(f1_(f0_))`
        >>> c["c"]()
        2

    """

    def __getitem__(self, key):
        functions: List = list(reversed(self.data[key]))
        f = functions[0]
        for fi in functions[1:]:
            f = fi(f)
        return f


def from_experimentalist_pipeline(pipeline: Pipeline) -> Executor:
    """Interface for running the experimentalist pipeline."""

    def _executor_experimentalist(state: SupportsControllerState, params: Dict):
        params_ = resolve_state_params(params, state)
        new_conditions = pipeline(**params_)

        if isinstance(new_conditions, pd.DataFrame):
            new_conditions_array = new_conditions
        elif isinstance(new_conditions, np.ndarray):
            _logger.warning(
                f"{new_conditions=} is an ndarray, so variable confusion is a possibility"
            )
            new_conditions_array = new_conditions
        elif isinstance(new_conditions, np.recarray):
            new_conditions_array = new_conditions
        elif isinstance(new_conditions, Iterable):
            # If the pipeline gives us an iterable, we need to make it into a concrete array.
            # We can't move this logic to the Pipeline, because the pipeline doesn't know whether
            # it's within another pipeline and whether it should convert the iterable to a
            # concrete array.
            new_conditions_values = list(new_conditions)
            new_conditions_array = np.array(new_conditions_values)
        else:
            raise NotImplementedError(
                f"Can't handle experimentalist output {new_conditions=}"
            )

        new_state = state.update(conditions=[new_conditions_array])
        return new_state

    return _executor_experimentalist


def from_experiment_runner_callable(callable: Callable) -> Executor:
    """Interface for running the experiment runner callable."""

    def _executor_experiment_runner(state: SupportsControllerState, params: Dict):
        params_ = resolve_state_params(params, state)
        x = state.conditions[-1]
        output = callable(x, **params_)

        if isinstance(x, pd.DataFrame):
            new_observations = output
        elif isinstance(x, np.ndarray):
            new_observations = np.column_stack([x, output])
        else:
            raise NotImplementedError(f"type {x=} not supported")

        new_state = state.update(observations=[new_observations])
        return new_state

    return _executor_experiment_runner


def from_theorist_estimator(estimator: BaseEstimator) -> Executor:
    """Interface for running the theorist estimator given some State."""

    def _executor_theorist(state: SupportsControllerState, params: Dict):
        params_ = resolve_state_params(params, state)
        variables = state.variables
        observations = state.observations
        assert (
            len(observations) >= 1
        ), f"{observations=} needs at least one entry for model fitting"

        if isinstance(observations[-1], pd.DataFrame):
            all_observations = pd.concat(observations)
            iv_names = [iv.name for iv in variables.independent_variables]
            dv_names = [dv.name for dv in variables.dependent_variables]
            x, y = all_observations[iv_names], all_observations[dv_names]
        elif isinstance(observations[-1], np.ndarray):
            all_observations = np.row_stack(observations)
            n_xs = len(variables.independent_variables)
            x, y = all_observations[:, :n_xs], all_observations[:, n_xs:]
            if y.shape[1] == 1:
                y = y.ravel()
        else:
            raise NotImplementedError(f"type {observations[-1]=} not supported")

        new_theorist = copy.deepcopy(estimator)
        new_theorist.fit(x, y, **params_)

        try:
            _logger.debug(
                f"fitted {new_theorist=}\nnew_theorist.__dict__:"
                f"\n{pprint.pformat(new_theorist.__dict__)}"
            )
        except AttributeError:
            _logger.debug(
                f"fitted {new_theorist=} "
                f"new_theorist has no __dict__ attribute, so no results are shown"
            )

        new_state = state.update(models=[new_theorist])
        return new_state

    return _executor_theorist


def full_cycle_wrapper(
    experimentalist_pipeline: Pipeline,
    experiment_runner_callable: Callable,
    theorist_estimator: BaseEstimator,
) -> Executor:
    """Interface for running the full AER cycle."""

    def _executor_full_cycle(
        state: SupportsControllerState, params: Dict
    ):  # TODO fix type
        experimentalist_params = params.get("experimentalist", {})

        experimentalist_executor = from_experimentalist_pipeline(
            experimentalist_pipeline
        )
        experimentalist_result = experimentalist_executor(state, experimentalist_params)

        experiment_runner_params = params.get("experiment_runner", {})
        experiment_runner_executor = from_experiment_runner_callable(
            experiment_runner_callable
        )
        experiment_runner_result = experiment_runner_executor(
            experimentalist_result, experiment_runner_params
        )

        theorist_params = params.get("theorist", {})
        theorist_executor = from_theorist_estimator(theorist_estimator)
        theorist_result = theorist_executor(experiment_runner_result, theorist_params)
        return theorist_result

    return _executor_full_cycle


def no_op(state, params):
    """
    An Executor which has no effect on the state.

    Examples:
         >>> from autora.workflow.state import Snapshot
         >>> s = Snapshot()
         >>> s_returned = no_op(s, {})
         >>> assert s_returned is s
    """
    _logger.warning("You called a `no_op` Executor. Returning the state unchanged.")
    return state
