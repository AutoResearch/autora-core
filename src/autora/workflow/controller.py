"""  The cycle controller for AER. """
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from autora.experimentalist.pipeline import Pipeline
from autora.variable import VariableCollection
from sklearn.base import BaseEstimator

from .base import BaseController
from .executor import (
    ChainedFunctionMapping,
    from_experiment_runner_callable,
    from_experimentalist_pipeline,
    from_theorist_estimator,
)
from .planner import last_result_kind_planner
from .state import History

_logger = logging.getLogger(__name__)


class Controller(BaseController[History]):
    """
    Runs an experimentalist, experiment runner, and theorist in order.

    Once initialized, the `controller` can be started by calling `next(controller)` or using the
        `controller.run` method. Each iteration runs the next logical step based on the last
        result:
    – if the last result doesn't exist or is a model, run the experimentalist and add an
        experimental condition as a new result,
    - if the last result is an experimental condition, run the experiment runner and add an
       observation as a new result,
    - if the last result is an observation, run the theorist and add a model as a new result.

    """

    def __init__(
        self,
        variables: Optional[VariableCollection] = None,
        theorist: Optional[BaseEstimator] = None,
        experimentalist: Optional[Pipeline] = None,
        experiment_runner: Optional[Callable] = None,
        params: Optional[Dict] = None,
        monitor: Optional[Callable[[History], None]] = None,
        planner: Callable[[History], str] = last_result_kind_planner,
    ):
        """
        Args:
            variables: a description of the dependent and independent variables
            theorist: a scikit-learn-compatible estimator
            experimentalist: an autora.experimentalist.Pipeline
            experiment_runner: a function to map independent variables onto observed dependent
                variables
            monitor: a function which gets read-only access to the `data` attribute at the end of
                each cycle.
            params: a nested dictionary with parameters to be passed to the parts of the cycle.
                E.g. if the experimentalist had a step named "pool" which took an argument "n",
                which you wanted to set to the value 30, then params would be set to this:
                `{"experimentalist": {"pool": {"n": 30}}}`
            planner: a function which maps from the state to the next ExecutorName. The default
                is to map from the last result in the state's history to the next logical step.
        """

        state = History(
            variables=variables,
            params=params,
            conditions=[],
            observations=[],
            models=[],
        )

        executor_collection = ChainedFunctionMapping(
            experimentalist=[from_experimentalist_pipeline, experimentalist],
            experiment_runner=[from_experiment_runner_callable, experiment_runner],
            theorist=[from_theorist_estimator, theorist],
        )

        super().__init__(
            state=state,
            planner=planner,
            executor_collection=executor_collection,
            monitor=monitor,
        )

    def seed(self, **kwargs):
        for key, value in kwargs.items():
            self.state = self.state.update(**{key: value})
