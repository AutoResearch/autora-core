""" Classes for storing and passing a cycle's state as an immutable snapshot. """
from dataclasses import dataclass, field
from typing import Dict, List

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.state.delta import Delta
from autora.state.protocol import SupportsControllerStateFields
from autora.variable import VariableCollection


@dataclass(frozen=True)
class Snapshot(SupportsControllerStateFields):
    """An object passed between and updated by processing steps in the Controller."""

    # Single values
    variables: VariableCollection = field(default_factory=VariableCollection)
    params: Dict = field(default_factory=dict)

    # Sequences
    conditions: List[ArrayLike] = field(default_factory=list)
    observations: List[ArrayLike] = field(default_factory=list)
    models: List[BaseEstimator] = field(default_factory=list)

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
            AssertionError: 'm1' must be a list, e.g. `['m1']`?)

        """

        def _coalesce_lists(old, new):
            assert isinstance(
                old, List
            ), f"{repr(old)} must be a list, e.g. `[{repr(old)}]`?)"
            if new is not None:
                assert isinstance(
                    new, List
                ), f"{repr(new)} must be a list, e.g. `[{repr(new)}]`?)"
                return old + list(new)
            else:
                return old

        variables_ = variables or self.variables
        params_ = params or self.params
        conditions_ = _coalesce_lists(self.conditions, conditions)
        observations_ = _coalesce_lists(self.observations, observations)
        models_ = _coalesce_lists(self.models, models)
        return Snapshot(variables_, params_, conditions_, observations_, models_)

    def __add__(self, other: Delta):
        """
        Add a delta to the object.

        Examples:
            The initial object is empty:
            >>> s0 = Snapshot()
            >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(variables=VariableCollection(...), params={}, conditions=[],
                            observations=[], models=[])

            We can update the params using the `+` operator:
            >>> from autora.state.delta import Delta
            >>> s0 + Delta(params={'first': 'params'})  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(..., params={'first': 'params'}, ...)

            ... but the original object is unchanged:
            >>> s0  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(..., params={}, ...)

            For params, only one object is returned from the respective property:
            >>> (s0 + Delta(params={'first': 'params'}) + Delta(params={'second':'params'})).params
            {'second': 'params'}

            ... and the same applies to variables:
            >>> from autora.variable import VariableCollection, IV
            >>> (s0 + Delta(variables=VariableCollection([IV("1st IV")])) +
            ...    Delta(variables=VariableCollection([IV("2nd IV")]))).variables
            VariableCollection(independent_variables=[IV(name='2nd IV',...)], ...)

            When we update the conditions, observations or models, the respective list is extended:
            >>> s3 = s0 + Delta(models=["1st model"])
            >>> s3
            Snapshot(..., models=['1st model'])

            ... so we can see the history of all the models, for instance.
            >>> s3 + Delta(models=["2nd model"])
            Snapshot(..., models=['1st model', '2nd model'])

            The same applies to observations:
            >>> s4 = s0 + Delta(observations=["1st observation"])
            >>> s4
            Snapshot(..., observations=['1st observation'], ...)

            >>> s4 + Delta(observations=["2nd observation"])  # doctest: +ELLIPSIS
            Snapshot(..., observations=['1st observation', '2nd observation'], ...)


            The same applies to conditions:
            >>> s5 = s0 + Delta(conditions=["1st condition"])
            >>> s5
            Snapshot(..., conditions=['1st condition'], ...)

            >>> s5 + Delta(conditions=["2nd condition"])  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Snapshot(..., conditions=['1st condition', '2nd condition'], ...)

            You can also update with multiple conditions, observations and models:
            >>> s0 + Delta(conditions=['c1', 'c2'])
            Snapshot(..., conditions=['c1', 'c2'], ...)

            >>> s0 + Delta(models=['m1', 'm2'], variables={'m': 1})
            Snapshot(variables={'m': 1}, ..., models=['m1', 'm2'])

            >>> s0 + Delta(models=['m1'], observations=['o1'], variables={'m': 1})
            Snapshot(variables={'m': 1}, ..., observations=['o1'], models=['m1'])


            Inputs to models, observations and conditions must be Lists
            which can be cast to lists:
            >>> s0 + Delta(models='m1')  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            AssertionError: 'm1' must be a list, e.g. `['m1']`?)
        """
        return self.update(**other)
