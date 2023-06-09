""" Classes for storing and passing a cycle's state as an immutable history. """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autora.state.delta import Delta
from autora.state.protocol import (
    ResultKind,
    SupportsControllerStateHistory,
    SupportsDataKind,
)
from autora.state.snapshot import Snapshot
from autora.variable import VariableCollection


class History(SupportsControllerStateHistory):
    """
    An immutable object for tracking the state and history of an AER cycle.
    """

    def __init__(
        self,
        variables: Optional[VariableCollection] = None,
        params: Optional[Dict] = None,
        conditions: Optional[List[ArrayLike]] = None,
        observations: Optional[List[ArrayLike]] = None,
        models: Optional[List[BaseEstimator]] = None,
        history: Optional[Sequence[Result]] = None,
    ):
        """

        Args:
            variables: a single datum to be marked as "variables"
            params: a single datum to be marked as "params"
            conditions: an iterable of data, each to be marked as "conditions"
            observations: an iterable of data, each to be marked as "observations"
            models: an iterable of data, each to be marked as "models"
            history: an iterable of Result objects to be used as the initial history.

        Examples:
            Empty input leads to an empty state:
            >>> History()
            History([])

            ... or with values for any or all of the parameters:
            >>> from autora.variable import VariableCollection
            >>> History(variables=VariableCollection()) # doctest: +ELLIPSIS
            History([Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)])

            >>> History(params={"some": "params"})
            History([Result(data={'some': 'params'}, kind=ResultKind.PARAMS)])

            >>> History(conditions=["a condition"])
            History([Result(data='a condition', kind=ResultKind.CONDITION)])

            >>> History(observations=["an observation"])
            History([Result(data='an observation', kind=ResultKind.OBSERVATION)])

            >>> from sklearn.linear_model import LinearRegression
            >>> History(models=[LinearRegression()])
            History([Result(data=LinearRegression(), kind=ResultKind.MODEL)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `variables`, `params`, `conditions`, `observations`, `models`
            >>> History(models=['m1', 'm2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data='from history', kind=ResultKind.VARIABLES),
                                    Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
                                    Result(data={'a': 'param'}, kind=ResultKind.PARAMS),
                                    Result(data='c1', kind=ResultKind.CONDITION),
                                    Result(data='c2', kind=ResultKind.CONDITION),
                                    Result(data='o1', kind=ResultKind.OBSERVATION),
                                    Result(data='o2', kind=ResultKind.OBSERVATION),
                                    Result(data='m1', kind=ResultKind.MODEL),
                                    Result(data='m2', kind=ResultKind.MODEL)])
        """
        self.data: List

        if history is not None:
            self.data = list(history)
        else:
            self.data = []

        self.data += _init_result_list(
            variables=variables,
            params=params,
            conditions=conditions,
            observations=observations,
            models=models,
        )

    def update(
        self,
        variables=None,
        params=None,
        conditions=None,
        observations=None,
        models=None,
        history=None,
    ):
        """
        Create a new object with updated values.

        Examples:
            The initial object is empty:
            >>> h0 = History()
            >>> h0
            History([])

            We can update the variables using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> h1 = h0.update(variables=VariableCollection())
            >>> h1  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)])

            ... the original object is unchanged:
            >>> h0
            History([])

            We can update the variables again:
            >>> h2 = h1.update(variables=VariableCollection(["some IV"]))
            >>> h2._by_kind  # doctest: +ELLIPSIS
            Snapshot(variables=VariableCollection(independent_variables=['some IV'],...), ...)

            ... and we see that there is only ever one variables object returned.

            Params is treated the same way as variables:
            >>> hp = h0.update(params={'first': 'params'})
            >>> hp
            History([Result(data={'first': 'params'}, kind=ResultKind.PARAMS)])

            ... where only the most recent "params" object is returned from the `.params` property.
            >>> hp = hp.update(params={'second': 'params'})
            >>> hp.params
            {'second': 'params'}

            ... however, the full history of the params objects remains available, if needed:
            >>> hp  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'first': 'params'}, kind=ResultKind.PARAMS),
                                    Result(data={'second': 'params'}, kind=ResultKind.PARAMS)])

            When we update the conditions, observations or models, a new entry is added to the
            history:
            >>> h3 = h0.update(models=["1st model"])
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st model', kind=ResultKind.MODEL)])

            ... so we can see the history of all the models, for instance.
            >>> h3 = h3.update(models=["2nd model"])  # doctest: +NORMALIZE_WHITESPACE
            >>> h3  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st model', kind=ResultKind.MODEL),
                                    Result(data='2nd model', kind=ResultKind.MODEL)])

            ... and the full history of models is available using the `.models` parameter:
            >>> h3.models
            ['1st model', '2nd model']

            The same for the observations:
            >>> h4 = h0.update(observations=["1st observation"])
            >>> h4
            History([Result(data='1st observation', kind=ResultKind.OBSERVATION)])

            >>> h4.update(observations=["2nd observation"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            History([Result(data='1st observation', kind=ResultKind.OBSERVATION),
                                    Result(data='2nd observation', kind=ResultKind.OBSERVATION)])


            The same for the conditions:
            >>> h5 = h0.update(conditions=["1st condition"])
            >>> h5
            History([Result(data='1st condition', kind=ResultKind.CONDITION)])

            >>> h5.update(conditions=["2nd condition"])  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='1st condition', kind=ResultKind.CONDITION),
                                    Result(data='2nd condition', kind=ResultKind.CONDITION)])

            You can also update with multiple conditions, observations and models:
            >>> h0.update(conditions=['c1', 'c2'])  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='c1', kind=ResultKind.CONDITION),
                                    Result(data='c2', kind=ResultKind.CONDITION)])

            >>> h0.update(models=['m1', 'm2'], variables={'m': 1}
            ... ) # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                     Result(data='m1', kind=ResultKind.MODEL),
                     Result(data='m2', kind=ResultKind.MODEL)])

            >>> h0.update(models=['m1'], observations=['o1'], variables={'m': 1}
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='m1', kind=ResultKind.MODEL)])

            We can also update with a complete history:
            >>> History().update(history=[Result(data={'m': 2}, kind=ResultKind.VARIABLES),
            ...                           Result(data='o1', kind=ResultKind.OBSERVATION),
            ...                           Result(data='m1', kind=ResultKind.MODEL)],
            ...                  conditions=['c1']
            ... )  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'m': 2}, kind=ResultKind.VARIABLES),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='m1', kind=ResultKind.MODEL),
                     Result(data='c1', kind=ResultKind.CONDITION)])

        """

        if history is not None:
            history_extension = history
        else:
            history_extension = []

        history_extension += _init_result_list(
            variables=variables,
            params=params,
            conditions=conditions,
            observations=observations,
            models=models,
        )
        new_full_history = self.data + history_extension

        return History(history=new_full_history)

    def __add__(self, other: Delta):
        """The initial object is empty:
        >>> h0 = History()
        >>> h0
        History([])

        We can update the variables using the `.update` method:
        >>> from autora.variable import VariableCollection
        >>> h1 = h0 + Delta(variables=VariableCollection())
        >>> h1  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        History([Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)])

        ... the original object is unchanged:
        >>> h0
        History([])

        We can update the variables again:
        >>> h2 = h1 + Delta(variables=VariableCollection(["some IV"]))
        >>> h2._by_kind  # doctest: +ELLIPSIS
        Snapshot(variables=VariableCollection(independent_variables=['some IV'],...), ...)

        ... and we see that there is only ever one variables object returned.

        Params is treated the same way as variables:
        >>> hp = h0 + Delta(params={'first': 'params'})
        >>> hp
        History([Result(data={'first': 'params'}, kind=ResultKind.PARAMS)])

        ... where only the most recent "params" object is returned from the `.params` property.
        >>> hp = hp + Delta(params={'second': 'params'})
        >>> hp.params
        {'second': 'params'}

        ... however, the full history of the params objects remains available, if needed:
        >>> hp  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data={'first': 'params'}, kind=ResultKind.PARAMS),
                                Result(data={'second': 'params'}, kind=ResultKind.PARAMS)])

        When we update the conditions, observations or models, a new entry is added to the
        history:
        >>> h3 = h0 + Delta(models=["1st model"])
        >>> h3  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data='1st model', kind=ResultKind.MODEL)])

        ... so we can see the history of all the models, for instance.
        >>> h3 = h3 + Delta(models=["2nd model"])  # doctest: +NORMALIZE_WHITESPACE
        >>> h3  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data='1st model', kind=ResultKind.MODEL),
                                Result(data='2nd model', kind=ResultKind.MODEL)])

        ... and the full history of models is available using the `.models` parameter:
        >>> h3.models
        ['1st model', '2nd model']

        The same for the observations:
        >>> h4 = h0 + Delta(observations=["1st observation"])
        >>> h4
        History([Result(data='1st observation', kind=ResultKind.OBSERVATION)])

        >>> h4 + Delta(observations=["2nd observation"]
        ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        History([Result(data='1st observation', kind=ResultKind.OBSERVATION),
                                Result(data='2nd observation', kind=ResultKind.OBSERVATION)])


        The same for the conditions:
        >>> h5 = h0 + Delta(conditions=["1st condition"])
        >>> h5
        History([Result(data='1st condition', kind=ResultKind.CONDITION)])

        >>> h5 + Delta(conditions=["2nd condition"])  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data='1st condition', kind=ResultKind.CONDITION),
                                Result(data='2nd condition', kind=ResultKind.CONDITION)])

        You can also update with multiple conditions, observations and models:
        >>> h0 + Delta(conditions=['c1', 'c2'])  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data='c1', kind=ResultKind.CONDITION),
                                Result(data='c2', kind=ResultKind.CONDITION)])

        >>> h0 + Delta(models=['m1', 'm2'], variables={'m': 1}
        ... ) # doctest: +NORMALIZE_WHITESPACE
        History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                 Result(data='m1', kind=ResultKind.MODEL),
                 Result(data='m2', kind=ResultKind.MODEL)])

        >>> h0 + Delta(models=['m1'], observations=['o1'], variables={'m': 1}
        ... )  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data={'m': 1}, kind=ResultKind.VARIABLES),
                 Result(data='o1', kind=ResultKind.OBSERVATION),
                 Result(data='m1', kind=ResultKind.MODEL)])

        We can also update with a complete history:
        >>> History() + Delta(history=[Result(data={'m': 2}, kind=ResultKind.VARIABLES),
        ...                           Result(data='o1', kind=ResultKind.OBSERVATION),
        ...                           Result(data='m1', kind=ResultKind.MODEL)],
        ...                  conditions=['c1']
        ... )  # doctest: +NORMALIZE_WHITESPACE
        History([Result(data={'m': 2}, kind=ResultKind.VARIABLES),
                 Result(data='o1', kind=ResultKind.OBSERVATION),
                 Result(data='m1', kind=ResultKind.MODEL),
                 Result(data='c1', kind=ResultKind.CONDITION)])
        """
        return self.update(**other)

    def __repr__(self):
        return f"{type(self).__name__}({self.history})"

    @property
    def _by_kind(self):
        return _history_to_kind(self.data)

    @property
    def variables(self) -> VariableCollection:
        """

        Examples:
            The initial object is empty:
            >>> h = History()

            ... and returns an emtpy variables object
            >>> h.variables
            VariableCollection(independent_variables=[], dependent_variables=[], covariates=[])

            We can update the variables using the `.update` method:
            >>> from autora.variable import VariableCollection
            >>> h = h.update(variables=VariableCollection(independent_variables=['some IV']))
            >>> h.variables  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some IV'], ...)

            We can update the variables again:
            >>> h = h.update(variables=VariableCollection(["some other IV"]))
            >>> h.variables  # doctest: +ELLIPSIS
            VariableCollection(independent_variables=['some other IV'], ...)

            ... and we see that there is only ever one variables object returned."""
        return self._by_kind.variables

    @property
    def params(self) -> Dict:
        """

        Returns:

        Examples:
            Params is treated the same way as variables:
            >>> h = History()
            >>> h = h.update(params={'first': 'params'})
            >>> h.params
            {'first': 'params'}

            ... where only the most recent "params" object is returned from the `.params` property.
            >>> h = h.update(params={'second': 'params'})
            >>> h.params
            {'second': 'params'}

            ... however, the full history of the params objects remains available, if needed:
            >>> h  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data={'first': 'params'}, kind=ResultKind.PARAMS),
                                    Result(data={'second': 'params'}, kind=ResultKind.PARAMS)])
        """
        return self._by_kind.params

    @property
    def conditions(self) -> List[ArrayLike]:
        """
        Returns:

        Examples:
            View the sequence of models with one conditions:
            >>> h = History(conditions=[(1,2,3,)])
            >>> h.conditions
            [(1, 2, 3)]

            ... or more conditions:
            >>> h = h.update(conditions=[(4,5,6),(7,8,9)])  # doctest: +NORMALIZE_WHITESPACE
            >>> h.conditions
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

        """
        return self._by_kind.conditions

    @property
    def observations(self) -> List[ArrayLike]:
        """

        Returns:

        Examples:
            The sequence of all observations is returned
            >>> h = History(observations=["1st observation"])
            >>> h.observations
            ['1st observation']

            >>> h = h.update(observations=["2nd observation"])
            >>> h.observations  # doctest: +ELLIPSIS
            ['1st observation', '2nd observation']

        """
        return self._by_kind.observations

    @property
    def models(self) -> List[BaseEstimator]:
        """

        Returns:

        Examples:
            View the sequence of models with one model:
            >>> s = History(models=["1st model"])
            >>> s.models  # doctest: +NORMALIZE_WHITESPACE
            ['1st model']

            ... or more models:
            >>> s = s.update(models=["2nd model"])  # doctest: +NORMALIZE_WHITESPACE
            >>> s.models
            ['1st model', '2nd model']

        """
        return self._by_kind.models

    @property
    def history(self) -> List[Result]:
        """

        Examples:
            We initialze some history:
            >>> h = History(models=['m1', 'm2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'},
            ...     variables=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)])

            Parameters passed to the constructor are included in the history in the following order:
            `history`, `variables`, `params`, `conditions`, `observations`, `models`

            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [Result(data='from history', kind=ResultKind.VARIABLES),
             Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
             Result(data={'a': 'param'}, kind=ResultKind.PARAMS),
             Result(data='c1', kind=ResultKind.CONDITION),
             Result(data='c2', kind=ResultKind.CONDITION),
             Result(data='o1', kind=ResultKind.OBSERVATION),
             Result(data='o2', kind=ResultKind.OBSERVATION),
             Result(data='m1', kind=ResultKind.MODEL),
             Result(data='m2', kind=ResultKind.MODEL)]

            If we add a new value, like the params object, the updated value is added to the
            end of the history:
            >>> h = h.update(params={'new': 'param'})
            >>> h.history  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [..., Result(data={'new': 'param'}, kind=ResultKind.PARAMS)]

        """
        return self.data

    def filter_by(self, kind: Optional[Set[Union[str, ResultKind]]] = None) -> History:
        """
        Return a copy of the object with only data belonging to the specified kinds.

        Examples:
            >>> h = History(models=['m1', 'm2'], conditions=['c1', 'c2'],
            ...     observations=['o1', 'o2'], params={'a': 'param'},
            ...    variables=VariableCollection(),
            ...     history=[Result("from history", ResultKind.VARIABLES)])

            >>> h.filter_by(kind={"MODEL"})   # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='m1', kind=ResultKind.MODEL),
                                    Result(data='m2', kind=ResultKind.MODEL)])

            >>> h.filter_by(kind={ResultKind.OBSERVATION})  # doctest: +NORMALIZE_WHITESPACE
            History([Result(data='o1', kind=ResultKind.OBSERVATION),
                                    Result(data='o2', kind=ResultKind.OBSERVATION)])

            If we don't specify any filter criteria, we get the full history back:
            >>> h.filter_by()   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            History([Result(data='from history', kind=ResultKind.VARIABLES),
                     Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
                     Result(data={'a': 'param'}, kind=ResultKind.PARAMS),
                     Result(data='c1', kind=ResultKind.CONDITION),
                     Result(data='c2', kind=ResultKind.CONDITION),
                     Result(data='o1', kind=ResultKind.OBSERVATION),
                     Result(data='o2', kind=ResultKind.OBSERVATION),
                     Result(data='m1', kind=ResultKind.MODEL),
                     Result(data='m2', kind=ResultKind.MODEL)])

        """
        if kind is None:
            return self
        else:
            kind_ = {ResultKind(s) for s in kind}
            filtered_history = _filter_history(self.data, kind_)
            new_object = History(history=filtered_history)
            return new_object


@dataclass(frozen=True)
class Result(SupportsDataKind):
    """
    Container class for data and variables.

    Examples:
        >>> Result()
        Result(data=None, kind=None)

        >>> Result("a")
        Result(data='a', kind=None)

        >>> Result(None, "MODEL")
        Result(data=None, kind=ResultKind.MODEL)

        >>> Result(data="b")
        Result(data='b', kind=None)

        >>> Result("c", "OBSERVATION")
        Result(data='c', kind=ResultKind.OBSERVATION)
    """

    data: Optional[Any] = None
    kind: Optional[ResultKind] = None

    def __post_init__(self):
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", ResultKind(self.kind))


def _init_result_list(
    variables: Optional[VariableCollection] = None,
    params: Optional[Dict] = None,
    conditions: Optional[Iterable[ArrayLike]] = None,
    observations: Optional[Iterable[ArrayLike]] = None,
    models: Optional[Iterable[BaseEstimator]] = None,
) -> List[Result]:
    """
    Initialize a list of Result objects

    Returns:

    Args:
        variables: a single datum to be marked as "variables"
        params: a single datum to be marked as "params"
        conditions: an iterable of data, each to be marked as "conditions"
        observations: an iterable of data, each to be marked as "observations"
        models: an iterable of data, each to be marked as "models"

    Examples:
        Empty input leads to an empty state:
        >>> _init_result_list()
        []

        ... or with values for any or all of the parameters:
        >>> from autora.variable import VariableCollection
        >>> _init_result_list(variables=VariableCollection()) # doctest: +ELLIPSIS
        [Result(data=VariableCollection(...), kind=ResultKind.VARIABLES)]

        >>> _init_result_list(params={"some": "params"})
        [Result(data={'some': 'params'}, kind=ResultKind.PARAMS)]

        >>> _init_result_list(conditions=["a condition"])
        [Result(data='a condition', kind=ResultKind.CONDITION)]

        >>> _init_result_list(observations=["an observation"])
        [Result(data='an observation', kind=ResultKind.OBSERVATION)]

        >>> from sklearn.linear_model import LinearRegression
        >>> _init_result_list(models=[LinearRegression()])
        [Result(data=LinearRegression(), kind=ResultKind.MODEL)]

        The input arguments are added to the data in the order `variables`,
        `params`, `conditions`, `observations`, `models`:
        >>> _init_result_list(variables=VariableCollection(),
        ...                  params={"some": "params"},
        ...                  conditions=["a condition"],
        ...                  observations=["an observation", "another observation"],
        ...                  models=[LinearRegression()],
        ... ) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [Result(data=VariableCollection(...), kind=ResultKind.VARIABLES),
         Result(data={'some': 'params'}, kind=ResultKind.PARAMS),
         Result(data='a condition', kind=ResultKind.CONDITION),
         Result(data='an observation', kind=ResultKind.OBSERVATION),
         Result(data='another observation', kind=ResultKind.OBSERVATION),
         Result(data=LinearRegression(), kind=ResultKind.MODEL)]

    """
    data = []

    if variables is not None:
        data.append(Result(variables, ResultKind.VARIABLES))

    if params is not None:
        data.append(Result(params, ResultKind.PARAMS))

    for seq, kind in [
        (conditions, ResultKind.CONDITION),
        (observations, ResultKind.OBSERVATION),
        (models, ResultKind.MODEL),
    ]:
        if seq is not None:
            for i in seq:
                data.append(Result(i, kind=kind))

    return data


def _history_to_kind(history: Sequence[Result]) -> Snapshot:
    """
    Convert a sequence of results into a Snapshot instance:

    Examples:
        History might be empty
        >>> history_ = []
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(variables=VariableCollection(...), params={},
                        conditions=[], observations=[], models=[])

        ... or with values for any or all of the parameters:
        >>> history_ = _init_result_list(params={"some": "params"})
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, ...)

        >>> history_ += _init_result_list(conditions=["a condition"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, conditions=['a condition'], ...)

        >>> _history_to_kind(history_).params
        {'some': 'params'}

        >>> history_ += _init_result_list(observations=["an observation"])
        >>> _history_to_kind(history_) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, conditions=['a condition'],
                        observations=['an observation'], ...)

        >>> from sklearn.linear_model import LinearRegression
        >>> history_ = [Result(LinearRegression(), kind=ResultKind.MODEL)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., models=[LinearRegression()])

        >>> from autora.variable import VariableCollection, IV
        >>> variables = VariableCollection(independent_variables=[IV(name="example")])
        >>> history_ = [Result(variables, kind=ResultKind.VARIABLES)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(variables=VariableCollection(independent_variables=[IV(name='example', ...

        >>> history_ = [Result({'some': 'params'}, kind=ResultKind.PARAMS)]
        >>> _history_to_kind(history_) # doctest: +ELLIPSIS
        Snapshot(..., params={'some': 'params'}, ...)

    """
    namespace = Snapshot(
        variables=_get_last_data_with_default(
            history, kind={ResultKind.VARIABLES}, default=VariableCollection()
        ),
        params=_get_last_data_with_default(
            history, kind={ResultKind.PARAMS}, default={}
        ),
        observations=_list_data(
            _filter_history(history, kind={ResultKind.OBSERVATION})
        ),
        models=_list_data(_filter_history(history, kind={ResultKind.MODEL})),
        conditions=_list_data(_filter_history(history, kind={ResultKind.CONDITION})),
    )
    return namespace


def _list_data(data: Sequence[SupportsDataKind]):
    """
    Extract the `.data` attribute of each item in a sequence, and return as a list.

    Examples:
        >>> _list_data([])
        []

        >>> _list_data([Result("a"), Result("b")])
        ['a', 'b']
    """
    return list(r.data for r in data)


def _filter_history(data: Iterable[SupportsDataKind], kind: Set[ResultKind]):
    return filter(lambda r: r.kind in kind, data)


def _get_last(data: Sequence[SupportsDataKind], kind: Set[ResultKind]):
    results_new_to_old = reversed(data)
    last_of_kind = next(_filter_history(results_new_to_old, kind=kind))
    return last_of_kind


def _get_last_data_with_default(data: Sequence[SupportsDataKind], kind, default):
    try:
        result = _get_last(data, kind).data
    except StopIteration:
        result = default
    return result
