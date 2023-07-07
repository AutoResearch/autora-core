"""Utilities to wrap common theorist, experimentalist and experiment runners as `f(State)`.
so that $n$ processes $f_i$ on states $S$ can be represented as
$$f_n(...(f_1(f_0(S))))$$
"""
from __future__ import annotations

from typing import Callable, Iterable, TypeVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from autora.experimentalist.pipeline import Pipeline
from autora.state.delta import Delta, State, wrap_to_use_state
from autora.variable import VariableCollection

S = TypeVar("S")
X = TypeVar("X")
Y = TypeVar("Y")
XY = TypeVar("XY")
Executor = Callable[[State], State]


def theorist_from_estimator(estimator: BaseEstimator) -> Executor:
    """
    Convert a scikit-learn compatible estimator into a function on a `State` object.

    Supports passing additional `**kwargs` which are used to update the estimator's params
    before fitting.
    """

    @wrap_to_use_state
    def theorist(
        experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs
    ):
        ivs = [v.name for v in variables.independent_variables]
        dvs = [v.name for v in variables.dependent_variables]
        X, y = experiment_data[ivs], experiment_data[dvs]
        new_model = estimator.set_params(**kwargs).fit(X, y)
        return Delta(model=new_model)

    return theorist


def experiment_runner_from_x_to_y_function(f: Callable[[X], Y]) -> Executor:
    """Wrapper for experiment_runner of the form $f(x) \rarrow y$, where `f` returns just the $y$
    values"""

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        y = f(x, **kwargs)
        experiment_data = pd.DataFrame.merge(x, y, left_index=True, right_index=True)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def experiment_runner_from_x_to_xy_function(f: Callable[[X], XY]) -> Executor:
    """Wrapper for experiment_runner of the form $f(x) \rarrow (x,y)$, where `f`
    returns both $x$ and $y$ values in a complete dataframe."""

    @wrap_to_use_state
    def experiment_runner(conditions: pd.DataFrame, **kwargs):
        x = conditions
        experiment_data = f(x, **kwargs)
        return Delta(experiment_data=experiment_data)

    return experiment_runner


def experimentalist_from_pipeline(pipeline: Pipeline) -> Executor:
    """Wrapper for experimentalists of the form $f() \rarrow x$, where `f`
    returns both $x$ and $y$ values in a complete dataframe."""

    @wrap_to_use_state
    def experimentalist(params):
        conditions = pipeline(**params)
        if isinstance(conditions, (pd.DataFrame, np.ndarray, np.recarray)):
            conditions_ = conditions
        elif isinstance(conditions, Iterable):
            conditions_ = np.array(list(conditions))
        else:
            raise NotImplementedError("type `%s` is not supported" % (type(conditions)))
        return Delta(conditions=conditions_)

    return experimentalist
