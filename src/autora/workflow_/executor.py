"""Functions to represent processes $f_i$ on states $S$ as $f_n(...(f_1(f_0(S))))$"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional

import pandas as pd
from autora.variable import Variable, VariableCollection
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from autora.workflow_.state import BaseState, Delta, S


def wrap_to_use_state(f):
    """

    Args:
        f:

    Returns:

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class S(BaseState):
        ...     conditions: list[int] = field(metadata={"delta": "replace"})

        >>> @wrap_to_use_state
        ... def function(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return Delta(conditions=new_conditions)

        >>> function(S(conditions=[1,2,3,4]))
        S(conditions=[11, 12, 13, 14])

        >>> function(S(conditions=[101,102,103,104]))
        S(conditions=[111, 112, 113, 114])

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        >>> @wrap_to_use_state
        ... def function(experimental_data: pd.DataFrame, variables: VariableCollection):
        ...     ivs = [v.name for v in variables.independent_variables]
        ...     dvs = [v.name for v in variables.dependent_variables]
        ...     X, y = experimental_data[ivs], experimental_data[dvs]
        ...     new_model = LinearRegression(fit_intercept=True).fit(X, y)
        ...     return Delta(model=new_model)

        >>> @dataclass(frozen=True)
        ... class T(BaseState):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experimental_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> t = T(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experimental_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )
        >>> t_prime = function(t)
        >>> t_prime.model.coef_, t_prime.model.intercept_
        (array([[1.]]), array([2.]))
    """
    parameters = inspect.signature(f).parameters

    @wraps(f)
    # TODO: how do we handle the params here?
    def _f(state: S, params: Optional[dict] = None) -> S:
        # Convert the dataclass to a dict of parameters
        arguments = dict((p, getattr(state, p)) for p in parameters)
        delta = f(**arguments)
        new_state = state + delta
        return new_state

    return _f
