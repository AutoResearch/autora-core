from collections import UserList
from functools import singledispatchmethod
from typing import List, Union

from autora.state import Delta
from autora.state_history_dataclass import Entry


class HistoryStateList(UserList):
    """

    Examples:
        >>> a = HistoryStateList()
        >>> a
        []

        >>> a + {"conditions": [1, 2, 3]}
        [Entry(kind='conditions', data=[1, 2, 3])]

        >>> a + Delta(conditions=[1, 2, 3])
        [Entry(kind='conditions', data=[1, 2, 3])]

        >>> a + Entry(kind='conditions', data=[1, 2, 3])
        [Entry(kind='conditions', data=[1, 2, 3])]

        >>> a + [Entry(kind='conditions', data=[1, 2, 3])]
        [Entry(kind='conditions', data=[1, 2, 3])]

        >>> b = a + {"conditions": [1, 2, 3]}

        >>> b + {"conditions": [4, 5, 6]}   # doctest: +NORMALIZE_WHITESPACE
        [Entry(kind='conditions', data=[1, 2, 3]),
         Entry(kind='conditions', data=[4, 5, 6])]

        >>> b + [{"conditions": [7, 8]}, {"conditions": [9, 10]}]  # doctest: +NORMALIZE_WHITESPACE
        [Entry(kind='conditions', data=[1, 2, 3]),
         Entry(kind='conditions', data=[7, 8]),
         Entry(kind='conditions', data=[9, 10])]

        >>> list(b.filter_by("conditions"))
        [Entry(kind='conditions', data=[1, 2, 3])]

        >>> from autora.state import Delta
        >>> a + {"experiment_data": [101, 102, 103],
        ...      "model": "it's a trap!"}  # doctest: +NORMALIZE_WHITESPACE
        [Entry(kind='experiment_data', data=[101, 102, 103]),
         Entry(kind='model', data="it's a trap!")]

    """

    data: List[Entry]

    @singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)

    @__add__.register
    def _(self, other: Entry):
        new = self.__class__(self.data + [other])
        return new

    @__add__.register
    def _(self, other: Union[Delta, dict]):
        new_data = [Entry(kind, data) for kind, data in other.items()]
        new = self + new_data
        return new

    @__add__.register
    def _(self, other: list):
        new = self
        for entry in other:
            new = new + entry
        return new

    def filter_by(self, kind):
        filtered_data = filter(lambda d: d.kind == kind, self.data)
        return filtered_data

    def __setitem__(self, i, item):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def __delitem__(self, i):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def append(self, item):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def insert(self, i, item):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def pop(self, i=-1):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def remove(self, item):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def clear(self):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def reverse(self):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def sort(self, /, *args, **kwds):
        raise NotImplementedError("HistoryStateList does not support updating in place")

    def extend(self, other):
        raise NotImplementedError("HistoryStateList does not support updating in place")
