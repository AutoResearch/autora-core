"""State objects with a history."""

import operator
from collections import UserList
from dataclasses import dataclass, field, fields, replace
from functools import reduce
from typing import Iterable, List, Mapping, Sequence, Union

from autora.state import Delta, State


class History(UserList):
    """Stores a sequence of Mappings and States which can be
    Examples:
        >>> History()
        []

        >>> History("abcde")
        ['a', 'b', 'c', 'd', 'e']

        >>> h1 = History([
        ...     {'n': None},
        ...     {'n': 1},
        ...     {'n': 2, 'foo': 'bar', 'qux': 'thud'},
        ...     {'foo': 'bat'},
        ...     {'n': 3, 'foo': 'baz'},
        ...     {'n': 4, 'foo': 'baz', 'qux': 'nom'},
        ...     {'n': 5, 'foo': 'baz', 'qux': 'thud'},
        ...     {'qux': 'moo'},
        ...     {'n': 6, 'foo': 'bar'}])


        The individual operations can be applied alone:
        >>> h1.of("n")
        [None, 1, 2, 3, 4, 5, 6]

        >>> h1.where(foo="bar")
        [{'n': 2, 'foo': 'bar', 'qux': 'thud'}, {'n': 6, 'foo': 'bar'}]

        >>> h1.contains("qux")  # doctest: +NORMALIZE_WHITESPACE
        [{'n': 2, 'foo': 'bar', 'qux': 'thud'},
         {'n': 4, 'foo': 'baz', 'qux': 'nom'},
         {'n': 5, 'foo': 'baz', 'qux': 'thud'},
         {'qux': 'moo'}]

         >>> h1.up_to_last(foo="bat")
         [{'n': None}, {'n': 1}, {'n': 2, 'foo': 'bar', 'qux': 'thud'}, {'foo': 'bat'}]

        They can also be chained:
        >>> h1.contains("qux").of("n")
        [2, 4, 5]

        >>> h1.where(foo="bar").of("n")
        [2, 6]

        >>> import operator
        >>> h1.where(foo="bar").of("n").reduce(operator.add)
        8

        >>> import pandas as pd
        >>> h2 = History([
        ...     {"df": pd.DataFrame(), "meta": "raw"},
        ...     {"df": {"a": [1, 2, 3], "b": ["a", "b", "c"]}, "meta": "raw"},
        ...     {"df": {"a": [1, 2, 3], "b": ["a", pd.NA, "c"]}, "meta": "filtered"},
        ...     {"df": {"a": [4, 5, 6], "b": ["d", "e", "f"]}, "meta": "raw"},
        ...     {"qdf": {"x": ["⍺", "β", "ɣ"]}, "meta": "qc"},
        ...     {"df": {"a": [1, 3, 4, 5, 6], "b": ["a", "c", pd.NA, "e", pd.NA]},
        ...      "meta": "filtered"}])

        >>> def append_dfs(a, b):
        ...     return pd.concat((pd.DataFrame(a), pd.DataFrame(b)), ignore_index=True)
        >>> h2.where(meta="raw").of("df").reduce(append_dfs)
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        4  5  e
        5  6  f

        >>> h2.where(meta="filtered").of("df")[-1]
        {'a': [1, 3, 4, 5, 6], 'b': ['a', 'c', <NA>, 'e', <NA>]}

        >>> h2.up_to_last(meta="qc").where(meta="filtered").of("df")[-1]
        {'a': [1, 2, 3], 'b': ['a', <NA>, 'c']}

    """

    data: List[Union[Mapping, State]]

    def where(self, **kwargs):
        new = self.__class__(_history_where(self.data, **kwargs))
        return new

    def up_to_last(self, **kwargs):
        new = self.__class__(_history_up_to_last(self.data, **kwargs))
        return new

    def contains(self, *args):
        new = self.__class__(_history_contains(self.data, *args))
        return new

    def of(self, key):
        new = self.__class__(_history_of(self.data, key))
        return new

    def reduce(self, function):
        new = reduce(function, self)
        return new


@dataclass(frozen=True)
class DeltaHistory(State):
    """
    Base object for dataclasses which use the Delta mechanism and store the history which led to
    the creation of the current state.

    Examples:

        This object stores the history in a list...
        >>> DeltaHistory()
        DeltaHistory(history=[...])

        ... which is initialized with a reference to the object itself:
        >>> a = DeltaHistory()
        >>> a.history[0] is a
        True

        ... and the object in the history also has a full copy of the history at that point:
        >>> a.history[0].history[0] is a
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
        >>> reconstruct(c.history.up_to_last(foo="baz"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        DeltaHistory(history=[DeltaHistory(history=[...]),
                              {'n': 1},
                              {'n': 2, 'foo': 'bar', 'qux': 'thud'},
                              {'n': 3, 'foo': 'baz'}])


        ... or where `qux=thud`
        >>> reconstruct(c.history.up_to_last(qux="thud")) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        DeltaHistory(history=[DeltaHistory(history=[...]),
                              {'n': 1},
                              {'n': 2, 'foo': 'bar', 'qux': 'thud'}])


        >>> reconstruct(_filter_to_last(lambda e: not "foo" in e, c.history))
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
        >>> l + Delta(l=list("d")) + Delta(m=list("u"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
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

    history: History = field(default_factory=History)

    def __post_init__(self):
        if self.history == []:
            self.history.append(self)

    def __add__(self, other):
        new = super().__add__(other, warn_on_unused_fields=False)
        new = replace(new, history=self.history + [other])
        return new


def reconstruct(history: Iterable[Union[State, Delta]]):
    """
    Examples:
        >>> reconstruct([DeltaHistory()])
        DeltaHistory(history=[...])

        >>> reconstruct([DeltaHistory(), {"foo": "bar"}])
        DeltaHistory(history=[DeltaHistory(history=[...]), {'foo': 'bar'}])

        >>> reconstruct([DeltaHistory(), {"foo": "bar"}, {"baz": "bat"}])
        DeltaHistory(history=[DeltaHistory(history=[...]), {'foo': 'bar'}, {'baz': 'bat'}])

        >>> reconstruct([])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: reduce() of empty ... with no initial value

    """
    new = reduce(operator.add, history)
    return new


def _filter_to_last(condition, iterable):
    """
    Filter an iterable and return all entries until the last where a condition is True.

    Examples:
        >>> list(_filter_to_last(lambda x: x == 4, range(8)))
        [0, 1, 2, 3, 4]

        >>> i = [dict(n=1), dict(n=2), dict(n=3), dict(q="this"), dict(n=4)]
        >>> list(_filter_to_last(lambda x: x.get("q", None) == "this", i))
        [{'n': 1}, {'n': 2}, {'n': 3}, {'q': 'this'}]

        If none of the conditions match, the generator is emtpy:
        >>> list(_filter_to_last(lambda x: x == 9, range(8)))
        []


        We consider a more realistic case with heterogenous types in the list.
        >>> from dataclasses import dataclass, field
        >>> from typing import Optional
        >>> @dataclass(frozen=True)
        ... class NState(DeltaHistory):
        ...    n: Optional[int] = None
        >>> from autora.state import Delta
        >>> j = [dict(n=1), dict(n=2), Delta(n=3), dict(q="this"), NState(n=4), dict(n=5)]

        If we have a real state object in the list, there might be problems with a simple condition:
        >>> list(_filter_to_last(lambda x: x.get("q", None) == "this", j))
        Traceback (most recent call last):
        ...
        AttributeError: 'NState' object has no attribute 'get'

        In this case, a more complex condition is needed, for instance using exception handling:
        >>> def condition_with_exception_handling(entry):
        ...     try:
        ...         result = entry.get("q", None) == "this"
        ...         return result
        ...     except AttributeError:
        ...         return False
        >>> list(_filter_to_last(condition_with_exception_handling, j))
        [{'n': 1}, {'n': 2}, {'n': 3}, {'q': 'this'}]

        ... or with explicit support for each type:
        >>> from typing import Union
        >>> def condition_with_type_support_factory(key, value):
        ...     def condition(entry):
        ...         if isinstance(entry, Union[dict, Delta]):
        ...             result = entry.get(key, None) == value
        ...         elif isinstance(entry, State):
        ...             if hasattr(entry, key):
        ...                 result = getattr(entry, key) == value
        ...             else:
        ...                 result = False
        ...         else:
        ...             raise TypeError
        ...         return result
        ...     return condition
        >>> condition_with_type_support = condition_with_type_support_factory("q", "this")
        >>> list(_filter_to_last(condition_with_type_support, j))
        [{'n': 1}, {'n': 2}, {'n': 3}, {'q': 'this'}]

        >>> condition_with_type_support_n = condition_with_type_support_factory("n", 4)
        >>> list(_filter_to_last(condition_with_type_support_n, j))
        [{'n': 1}, {'n': 2}, {'n': 3}, {'q': 'this'}, NState(history=[...], n=4)]


    """
    reversed_iterable = reversed(iterable)
    reversed_index = len(iterable)
    for ri, e in enumerate(reversed_iterable):
        if condition(e):
            reversed_index = ri
            break

    if reversed_index is None:
        raise IndexError("no matching entries found")

    index = len(iterable) - reversed_index
    for i, e in enumerate(iterable[:index]):
        yield e


def _history_up_to_last(history: Sequence[Union[Mapping, State]], **kwargs):
    """

    Args:
        history:
        **kwargs:

    Returns:

    Examples:
        We consider a realistic case with heterogenous types in the list.
        >>> from dataclasses import dataclass, field
        >>> from typing import Optional
        >>> @dataclass(frozen=True)
        ... class NState(DeltaHistory):
        ...    n: Optional[int] = None
        >>> from autora.state import Delta
        >>> j = [dict(n=1), dict(n=2), Delta(n=3), dict(q="this"), NState(n=4), dict(n=5)]

        >>> list(_history_up_to_last(j, q="this"))
        [{'n': 1}, {'n': 2}, {'n': 3}, {'q': 'this'}]

        >>> list(_history_up_to_last(j, n=4))
        [{'n': 1}, {'n': 2}, {'n': 3}, {'q': 'this'}, NState(history=[...], n=4)]

        >>> list(_history_up_to_last([], n=4))
        []

    """

    def condition(entry):
        if isinstance(entry, Mapping):
            for key, value in kwargs.items():
                if not entry.get(key, None) == value:
                    return False
            return True
        elif isinstance(entry, State):
            for key, value in kwargs.items():
                if not (hasattr(entry, key) and getattr(entry, key) == value):
                    return False
            return True
        else:
            raise NotImplementedError("type %s not supported", type(entry))

    history = _filter_to_last(condition, history)
    return history


def _history_where(history: Sequence[Union[Mapping, State]], **kwargs):
    """
    Filter a history list for entries which match the given keyword arguments and their values

    Args:
        history: Iterable of Mappings (including Deltas) and States
        **kwargs: keys and their values to match

    Returns: filtered iterable of Mappings (including Deltas) and States

    Examples:
        >>> history = [
        ...     dict(a=1, b=2),
        ...     Delta(c=3, a=1),
        ...     Delta(c=3, a=2, unique=True),
        ...     DeltaHistory(),
        ...     dict()
        ... ]

        >>> list(_history_where(history))  # doctest: +NORMALIZE_WHITESPACE
        [{'a': 1, 'b': 2},
         {'c': 3, 'a': 1},
         {'c': 3, 'a': 2, 'unique': True},
         DeltaHistory(history=[...]),
         {}]

        >>> list(_history_where(history, a=1))
        [{'a': 1, 'b': 2}, {'c': 3, 'a': 1}]

        >>> list(_history_where(history, b=2))
        [{'a': 1, 'b': 2}]

        >>> list(_history_where(history, unique=True))
        [{'c': 3, 'a': 2, 'unique': True}]

        >>> list(_history_where(history, unique=1))
        [{'c': 3, 'a': 2, 'unique': True}]

    """

    def condition(entry):
        if isinstance(entry, Mapping):
            if kwargs.items() <= entry.items():
                return True
            else:
                return False
        elif isinstance(entry, State):
            entry_items = {
                k: hasattr(entry, k) and getattr(entry, k) for k in kwargs.keys()
            }
            if kwargs.items() <= entry_items.items():
                return True
            else:
                return False
        else:
            raise NotImplementedError("type %s not supported", type(entry))

    filtered_history = filter(condition, history)
    return filtered_history


def _history_contains(history: Sequence[Union[Mapping, State]], *args):
    """
    Filter a history list for entries which have the given keyword

    Args:
        history: Iterable of Mappings (including Deltas) and States
        *args: names of fields

    Returns: filtered iterable of Mappings (including Deltas) and States

    Examples:
        >>> history = [
        ...     dict(a=1, b=2),
        ...     Delta(c=3, a=1),
        ...     Delta(c=3, a=2, unique=True),
        ...     DeltaHistory(),
        ...     dict()
        ... ]

        >>> list(_history_contains(history))  # doctest: +NORMALIZE_WHITESPACE
        [{'a': 1, 'b': 2},
         {'c': 3, 'a': 1},
         {'c': 3, 'a': 2, 'unique': True},
         DeltaHistory(history=[...]),
         {}]

        >>> list(_history_contains(history, 'a'))
        [{'a': 1, 'b': 2}, {'c': 3, 'a': 1}, {'c': 3, 'a': 2, 'unique': True}]

        >>> list(_history_contains(history, 'b'))
        [{'a': 1, 'b': 2}]

        >>> list(_history_contains(history, 'c'))
        [{'c': 3, 'a': 1}, {'c': 3, 'a': 2, 'unique': True}]

        >>> list(_history_contains(history, 'unique'))
        [{'c': 3, 'a': 2, 'unique': True}]

    """
    keys = set(args)

    def condition(entry):
        if isinstance(entry, Mapping):
            if keys <= set(entry.keys()):
                return True
            else:
                return False
        elif isinstance(entry, State):
            if keys <= set(f.name for f in fields(entry)):
                return True
            else:
                return False

    filtered_history = filter(condition, history)
    return filtered_history


def _history_of(history: Iterable[Union[Mapping, State]], key: str):
    """
    Get all the values of a given key out of a history.

    Examples:

        >>> history = [
        ...     {'n': None},
        ...     {'n': 1},
        ...     {'n': 2, 'foo': 'bar', 'qux': 'thud'},
        ...     {'foo': 'bat'},
        ...     {'n': 3, 'foo': 'baz'},
        ...     {'n': 4, 'foo': 'baz', 'qux': 'nom'},
        ...     {'n': 5, 'foo': 'baz'},
        ...     {'qux': 'moo'},
        ...     {'n': 6, 'foo': 'bar'}]
        >>> list(_history_of(history, 'n'))
        [None, 1, 2, 3, 4, 5, 6]

        >>> list(_history_of(history, 'foo'))
        ['bar', 'bat', 'baz', 'baz', 'baz', 'bar']

        >>> list(_history_of(history, 'qux'))
        ['thud', 'nom', 'moo']

        >>> from dataclasses import dataclass, field
        >>> from typing import Optional
        >>> @dataclass(frozen=True)
        ... class NState(State):
        ...    n: Optional[int] = field(default=None, metadata={"delta": "replace"})
        >>> c = [NState(),
        ...      Delta(n=1),
        ...      Delta(n=2, foo="bar", qux="thud"),
        ...      {'foo': 'bat'},
        ...      Delta(n=3, foo="baz"),
        ...      Delta(n=4, foo="baz", qux="nom"),
        ...      {'qux': 'moo'},
        ...      Delta(n=5, foo="baz"),
        ...      Delta(n=6, foo="bar"),
        ... ]

        >>> list(_history_of(c, "n"))
        [None, 1, 2, 3, 4, 5, 6]

        >>> list(_history_of(c, "qux"))
        ['thud', 'nom', 'moo']

        >>> list(_history_of(c, "foo"))
        ['bar', 'bat', 'baz', 'baz', 'baz', 'bar']

    """

    for entry in history:
        if isinstance(entry, Mapping):
            if key in entry.keys():
                yield entry[key]
        elif isinstance(entry, State):
            if key in [f.name for f in fields(entry)]:
                yield getattr(entry, key)
