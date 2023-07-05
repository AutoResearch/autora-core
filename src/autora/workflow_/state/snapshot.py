from dataclasses import dataclass, field
from typing import Dict, List

from autora.variable import VariableCollection
from numpy._typing import ArrayLike
from sklearn.base import BaseEstimator

from .base import Delta, State


@dataclass(frozen=True)
class Snapshot(State):
    """An object passed between and updated by processing steps in the Controller."""

    # Single values
    variables: VariableCollection = field(
        default_factory=VariableCollection, metadata={"delta": "replace"}
    )
    params: Dict = field(default_factory=dict, metadata={"delta": "replace"})

    # Sequences
    conditions: List[ArrayLike] = field(
        default_factory=list, metadata={"delta": "extend"}
    )
    observations: List[ArrayLike] = field(
        default_factory=list, metadata={"delta": "extend"}
    )
    models: List[BaseEstimator] = field(
        default_factory=list, metadata={"delta": "extend"}
    )

    def update(
        self,
        variables=None,
        params=None,
        conditions=None,
        observations=None,
        models=None,
    ):
        """
        Create a new object with updated values.

        Examples:
            The initial object is empty:
            >>> s0 = Snapshot()
            >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(variables=VariableCollection(...), params={}, conditions=[],
                            observations=[], models=[])

            We can update the params using the `.update` method:
            >>> s0.update(params={'first': 'params'})  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(..., params={'first': 'params'}, ...)

            ... but the original object is unchanged:
            >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(..., params={}, ...)

            For params, only one object is returned from the respective property:
            >>> s0.update(params={'first': 'params'}).update(params={'second': 'params'}).params
            {'second': 'params'}

            ... and the same applies to variables:
            >>> from autora.variable import VariableCollection, IV
            >>> (s0.update(variables=VariableCollection([IV("1st IV")]))
            ...    .update(variables=VariableCollection([IV("2nd IV")]))).variables
            VariableCollection(independent_variables=[IV(name='2nd IV',...)], ...)

            When we update the conditions, observations or models, the respective list is extended:
            >>> s3 = s0.update(models=["1st model"])
            >>> s3
            Snapshot(..., models=['1st model'])

            ... so we can see the history of all the models, for instance.
            >>> s3.update(models=["2nd model"])
            Snapshot(..., models=['1st model', '2nd model'])

            The same applies to observations:
            >>> s4 = s0.update(observations=["1st observation"])
            >>> s4
            Snapshot(..., observations=['1st observation'], ...)

            >>> s4.update(observations=["2nd observation"])  # doctest: +ELLIPSIS
            Snapshot(..., observations=['1st observation', '2nd observation'], ...)


            The same applies to conditions:
            >>> s5 = s0.update(conditions=["1st condition"])
            >>> s5
            Snapshot(..., conditions=['1st condition'], ...)

            >>> s5.update(conditions=["2nd condition"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(..., conditions=['1st condition', '2nd condition'], ...)

            You can also update with multiple conditions, observations and models:
            >>> s0.update(conditions=['c1', 'c2'])
            Snapshot(..., conditions=['c1', 'c2'], ...)

            >>> s0.update(models=['m1', 'm2'], variables={'m': 1})
            Snapshot(variables={'m': 1}, ..., models=['m1', 'm2'])

            >>> s0.update(models=['m1'], observations=['o1'], variables={'m': 1})
            Snapshot(variables={'m': 1}, ..., observations=['o1'], models=['m1'])


            Inputs to models, observations and conditions must be Lists
            which can be cast to lists:
            >>> s0.update(models='m1')  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            TypeError: can only concatenate list (not "str") to list

        """

        delta = Delta(
            variables=variables,
            params=params,
            conditions=conditions,
            observations=observations,
            models=models,
        )
        return self + delta
