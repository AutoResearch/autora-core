import operator
from dataclasses import dataclass, field, replace
from functools import reduce
from typing import Iterable, List, Union

from autora.state import Delta, State


def reconstruct(history: Iterable[Union[State, Delta]]):
    """
    Examples:
        >>> reconstruct([DeltaHistory()])
        DeltaHistory(history=[...])

        >>> reconstruct([DeltaHistory(), {"foo": "bar"}])
        DeltaHistory(history=[DeltaHistory(history=[...]), {'foo': 'bar'}])

        >>> reconstruct([DeltaHistory(), {"foo": "bar"}, {"baz": "bat"}])
        DeltaHistory(history=[DeltaHistory(history=[...]), {'foo': 'bar'}, {'baz': 'bat'}])

        >>> reconstruct([])
        Traceback (most recent call last):
        ...
        TypeError: reduce() of empty iterable with no initial value

    """
    new = reduce(operator.add, history)
    return new


def filter_to_last(condition, history):
    reversed_history = reversed(history)
    reversed_index = None
    for ri, e in enumerate(reversed_history):
        if condition(e):
            reversed_index = ri
            break

    if reversed_index is None:
        raise IndexError("no matching entries found")

    index = len(history) - reversed_index
    for i, e in enumerate(history[:index]):
        yield e


@dataclass(frozen=True)
class DeltaHistory(State):
    """
    Base object for dataclasses which use the Delta mechanism and store the history which led to
    the creation of the current state.

    Examples:

        This object stores the history in a list...
        >>> DeltaHistory()
        DeltaHistory(history=[...])

        ... which is initialized with a reference to the object itself.
        >>> a = DeltaHistory()
        >>> a.history[0] is a
        True

        Each time a delta is added, the history is updated:
        >>> b = a + Delta(n=1) + Delta(n=2) + Delta(n=3) + Delta(n=4)
        >>> b
        DeltaHistory(history=[DeltaHistory(history=[...]), {'n': 1}, {'n': 2}, {'n': 3}, {'n': 4}])

        The history can be arbitrarily long:
        >>> q = DeltaHistory()
        >>> for i in range(10_000):
        ...     q += Delta(n=i)
        >>> q.history[-5:]
        [{'n': 9995}, {'n': 9996}, {'n': 9997}, {'n': 9998}, {'n': 9999}]

        Any fields which aren't present in the DeltaHistory object but which are provided in a
        Delta will be stored in the history
        >>> c: DeltaHistory = (a + Delta(n=1)
        ...      + Delta(n=2, foo="bar", qux="thud")
        ...      + Delta(n=3, foo="baz")
        ...      + Delta(n=4, foo="bar"))
        >>> c.history # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [DeltaHistory(history=[...]),
         {'n': 1},
         {'n': 2, 'foo': 'bar', 'qux': 'thud'},
         {'n': 3, 'foo': 'baz'},
         {'n': 4, 'foo': 'bar'}]

        We can reconstruct the history up to any point:
        >>> reconstruct(b.history[0:-1])
        DeltaHistory(history=[DeltaHistory(history=[...]), {'n': 1}, {'n': 2}, {'n': 3}])

        We can reconstruct the object up until the last entry where `foo=baz` (Note that we use
        the `e.get("foo", default)` method rather than the `e["foo"]` syntax
        so that if the Delta has no "foo" key, no error is thrown.):
        >>> reconstruct(filter_to_last(lambda e: e.get("foo", None) == "baz", c.history))
        ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        DeltaHistory(history=[DeltaHistory(history=[...]),
                              {'n': 1},
                              {'n': 2, 'foo': 'bar', 'qux': 'thud'},
                              {'n': 3, 'foo': 'baz'}])


        ... or where `qux=thud`
        >>> reconstruct(filter_to_last(lambda e: e.get("qux", None) == "thud", c.history))
        ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        DeltaHistory(history=[DeltaHistory(history=[...]),
                              {'n': 1},
                              {'n': 2, 'foo': 'bar', 'qux': 'thud'}])


        >>> reconstruct(filter_to_last(lambda e: not "foo" in e, c.history))
        ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        DeltaHistory(history=[DeltaHistory(history=[...]),
                              {'n': 1}])


        In real use, the DeltaHistory will be extended with additional fields.
        We define a dataclass where each field (which is going to be delta-ed) has additional
        metadata "delta" which describes its delta behaviour.
        >>> from dataclasses import dataclass, field
        >>> from typing import List, Optional
        >>> @dataclass(frozen=True)
        ... class ListState(DeltaHistory):
        ...    l: List = field(default_factory=list, metadata={"delta": "extend"})
        ...    m: List = field(default_factory=list, metadata={"delta": "replace"})

        If we instantiate the object with no data, the data fields will be initialized with
        default values and the history will be initialized with a reference to the returned object
        itself.
        >>> ListState()
        ListState(history=[...], l=[], m=[])

        If we instantiate the dataclass with some data:
        >>> l = ListState(l=list("abc"), m=list("xyz"))
        >>> l  # doctest: +NORMALIZE_WHITESPACE
        ListState(history=[...], l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ...the first reference in the history list is the object itself:
        >>> l.history[0]
        ListState(history=[...], l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... the two objects are identical.
        >>> l is l.history[0]
        True


        We can add deltas to it. `l` will be extended:
        >>> l + Delta(l=list("def"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(history=[..., {'l': ['d', 'e', 'f']}],
                  l=['a', 'b', 'c', 'd', 'e', 'f'],
                  m=['x', 'y', 'z'])

        ... wheras `m` will be replaced:
        >>> l + Delta(m=list("uvw"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(history=[..., {'m': ['u', 'v', 'w']}],
                  l=['a', 'b', 'c'],
                  m=['u', 'v', 'w'])

        ... they can be chained:
        >>> l +Delta(l=list("d")) + Delta(m=list("u"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(history=[..., {'l': ['d']}, {'m': ['u']}],
                  l=['a', 'b', 'c', 'd'],
                  m=['u'])

        ... and we update multiple fields with one Delta:
        >>> l + Delta(l=list("ghi"), m=list("rst"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(history=[..., {'l': ['g', 'h', 'i'], 'm': ['r', 's', 't']}],
                  l=['a', 'b', 'c', 'g', 'h', 'i'],
                  m=['r', 's', 't'])

        A non-existent field will be ignored but included in the history:
        >>> l + Delta(o="not a field")  # doctest: +ELLIPSIS
        ListState(history=[..., {'o': 'not a field'}], l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        We can also use the `.update` method to do the same thing:
        >>> l.update(l=list("ghi"), m=list("rst"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(history=[..., {'l': ['g', 'h', 'i'], 'm': ['r', 's', 't']}],
                  l=['a', 'b', 'c', 'g', 'h', 'i'],
                  m=['r', 's', 't'])

        We can also define fields which `append` the last result:
        >>> @dataclass(frozen=True)
        ... class AppendState(DeltaHistory):
        ...    n: List = field(default_factory=list, metadata={"delta": "append"})

        >>> m = AppendState(n=list("ɑβɣ"))
        >>> m
        AppendState(history=[...], n=['ɑ', 'β', 'ɣ'])

        `n` will be appended:
        >>> m + Delta(n="∂")  # doctest: +ELLIPSIS
        AppendState(history=[..., {'n': '∂'}], n=['ɑ', 'β', 'ɣ', '∂'])

        The metadata key "converter" is used to coerce types (inspired by
        [PEP 712](https://peps.python.org/pep-0712/)):
        >>> @dataclass(frozen=True)
        ... class CoerceStateList(DeltaHistory):
        ...    o: Optional[List] = field(default=None, metadata={"delta": "replace"})
        ...    p: List = field(default_factory=list, metadata={"delta": "replace",
        ...                                                    "converter": list})

        >>> r = CoerceStateList()

        If there is no `metadata["converter"]` set for a field, no coercion occurs
        >>> r + Delta(o="not a list")  # doctest: +ELLIPSIS
        CoerceStateList(history=[..., {'o': 'not a list'}], o='not a list', p=[])

        If there is a `metadata["converter"]` set for a field, the data are coerced,
        but the Delta object in the history remains the same as it was input:
        >>> r + Delta(p="not a list")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        CoerceStateList(history=[..., {'p': 'not a list'}],
                        o=None, p=['n', 'o', 't', ' ', 'a', ' ', 'l', 'i', 's', 't'])

        If the input data are of the correct type, they are returned unaltered:
        >>> r + Delta(p=["a", "list"])  # doctest: +ELLIPSIS
        CoerceStateList(history=[..., {'p': ['a', 'list']}], o=None, p=['a', 'list'])

        With a converter, inputs are converted to the type output by the converter:
        >>> import pandas as pd
        >>> @dataclass(frozen=True)
        ... class CoerceStateDataFrame(DeltaHistory):
        ...    q: pd.DataFrame = field(default_factory=pd.DataFrame,
        ...                            metadata={"delta": "replace",
        ...                                      "converter": pd.DataFrame})

        If the type is already correct, the object is passed to the converter,
        but should be returned unchanged:
        >>> s = CoerceStateDataFrame()
        >>> (s + Delta(q=pd.DataFrame([("a",1,"alpha"), ("b",2,"beta")], columns=list("xyz")))).q
           x  y      z
        0  a  1  alpha
        1  b  2   beta

        If the type is not correct, the object is converted if possible. For a dataframe,
        we can convert records:
        >>> (s + Delta(q=[("a",1,"alpha"), ("b",2,"beta")])).q
           0  1      2
        0  a  1  alpha
        1  b  2   beta

        ... but be aware that the history stores the Delta object as it was provided,
        before coercion.
        >>> (s + Delta(q=[("a",1,"alpha"), ("b",2,"beta")])).history  # doctest: +ELLIPSIS
        [..., {'q': [('a', 1, 'alpha'), ('b', 2, 'beta')]}]

    """

    history: List[Delta] = field(default_factory=list)

    def __add__(self, other):
        new = super().__add__(other, warn_on_unused_fields=False)
        new = replace(new, history=self.history + [other])
        return new

    def __post_init__(self):
        if self.history == []:
            self.history.append(self)

    def as_of_last(self, **kwargs):
        """
        Returns the State as it was the last time all the keyword-value pairs were found in the
        Delta.

        Examples:

            >>> from dataclasses import dataclass, field
            >>> @dataclass(frozen=True)
            ... class NState(DeltaHistory):
            ...    n: int = field(default=0, metadata={"delta": "replace"})
            >>> c = (NState()
            ...      + Delta(n=1)
            ...      + Delta(n=2, foo="bar", qux="thud")
            ...      + Delta(n=3, foo="baz")
            ...      + Delta(n=4, foo="baz", qux="nom")
            ...      + Delta(n=5, foo="baz")
            ...      + Delta(n=6, foo="bar"))

            The last time `foo` was equal to `"bar"` in one of the Delta updates was the last step:
            >>> c.as_of_last(foo="bar")  # doctest: +ELLIPSIS
            NState(history=[...], n=6)

            >>> c.as_of_last(foo="baz")  # doctest: +ELLIPSIS
            NState(history=[...], n=5)

            >>> c.as_of_last(qux="thud")  # doctest: +ELLIPSIS
            NState(history=[...], n=2)

            >>> c.as_of_last(foo="baz", qux="nom")  # doctest: +ELLIPSIS
            NState(history=[...], n=4)

            We can also look up values which might have been updated:
            >>> c.as_of_last(n=2)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            NState(history=[NState(history=[...], n=0),
                            {'n': 1},
                            {'n': 2, 'foo': 'bar', 'qux': 'thud'}],
                   n=2)


            This also works with extending and appending fields:
            >>> @dataclass(frozen=True)
            ... class EState(DeltaHistory):
            ...    n: list[int] = field(default_factory=list, metadata={"delta": "append"})
            >>> d = (EState()
            ...      + Delta(n=1)
            ...      + Delta(n=2, foo="bar", qux="thud")
            ...      + Delta(n=3, foo="baz")
            ...      + Delta(n=4, foo="baz", qux="nom")
            ...      + Delta(n=5, foo="baz")
            ...      + Delta(n=6, foo="bar"))

            The last time `foo` was equal to `"bar"` in one of the Delta updates was the last step:
            >>> d.as_of_last(foo="bar")  # doctest: +ELLIPSIS
            EState(history=[...], n=[1, 2, 3, 4, 5, 6])

            >>> d.as_of_last(foo="baz")  # doctest: +ELLIPSIS
            EState(history=[...], n=[1, 2, 3, 4, 5])

            >>> d.as_of_last(qux="thud")  # doctest: +ELLIPSIS
            EState(history=[...], n=[1, 2])

            >>> d.as_of_last(foo="baz", qux="nom")  # doctest: +ELLIPSIS
            EState(history=[...], n=[1, 2, 3, 4])

            We can also look up values which might have been updated:
            >>> d.as_of_last(n=3)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            EState(history=[EState(history=[...], n=[]),
                            {'n': 1},
                            {'n': 2, 'foo': 'bar', 'qux': 'thud'},
                            {'n': 3, 'foo': 'baz'}],
                   n=[1, 2, 3])

        """

        def condition(entry):
            for key, value in kwargs.items():
                if not entry.get(key, None) == value:
                    return False
            return True

        history = filter_to_last(condition, self.history)
        new = reconstruct(history)

        return new

    def history_of(self, key: str):
        relevant_history_entries = filter(
            lambda e: key in e.keys(),
            self.history,
        )
        relevant_entries = [k.get(key) for k in relevant_history_entries]
        return relevant_entries

    def history_filter(self, cond):
        relevant_history_entries = list(filter(cond, self.history))
        return relevant_history_entries
