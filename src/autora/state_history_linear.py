from collections import namedtuple
from dataclasses import dataclass, field, replace
from typing import Any, List

from autora.state import State

# This is an object held outside the state which allows us to track the history
# Each update adds another tuple to the list, and we can look through the list if we want to
# recover the history.
# Serializing will be a bit of a pain.
Relationship = namedtuple("Relationship", ["parent", "child"])
_history: List[Relationship] = list()


@dataclass(frozen=True)
class StateHistory(State):
    """
    Base object for dataclasses which use the Delta mechanism and have a history

    Examples:
        >>> from autora.state import Delta

        The history of the state objects can be returned from the `.history` method.
        >>> a = StateHistory()
        >>> a
        StateHistory(meta=None, parent=None)

        Each time a new delta is added to the object, a new state is returned with the original
        as its parent:
        >>> a + Delta(meta=1) # doctest: +NORMALIZE_WHITESPACE
        StateHistory(meta=1,
                     parent=StateHistory(meta=None, parent=None))

        ... and each parent includes all its parents
        >>> a + Delta(meta=1) + Delta(meta=2) + Delta(meta=3) # doctest: +NORMALIZE_WHITESPACE
        StateHistory(meta=3,
                     parent=StateHistory(meta=2,
                                         parent=StateHistory(meta=1,
                                                             parent=StateHistory(meta=None,
                                                                                parent=None))))

        These chains of parents can be as deep as the recursion limit allows:
        >>> b = StateHistory()
        >>> for i in range(1, 100):
        ...     b += Delta(meta=i)
        >>> b  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StateHistory(meta=99, parent=StateHistory(meta=98, parent=StateHistory(meta=97...

        The `.history` method returns the states from newest to oldest in a generator:
        >>> b.history  # doctest: +ELLIPSIS
        <generator object StateHistory.history at 0x...>

        >>> h = b.history
        >>> next(h)  # doctest: +ELLIPSIS
        StateHistory(meta=98, ...)

        >>> next(h)  # doctest: +ELLIPSIS
        StateHistory(meta=97, ...)

        If the original order is desired, then we can reverse the history by listing it fully and
        reversing that:
        >>> list(reversed(list(b.history)))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [StateHistory(meta=None, parent=None),
         StateHistory(meta=1, parent=StateHistory(meta=None, parent=None)),
         StateHistory(meta=2, parent=StateHistory(meta=1, ...))),
        ...]

        >>> from dataclasses import dataclass, field
        >>> from typing import List, Optional

        We define a dataclass where each field (which is going to be delta-ed) has additional
        metadata "delta" which describes its delta behaviour.
        >>> @dataclass(frozen=True)
        ... class ListState(StateHistory):
        ...    l: List = field(default_factory=list, metadata={"delta": "extend"})
        ...    m: List = field(default_factory=list, metadata={"delta": "replace"})

        >>> ListState()
        ListState(meta=None, parent=None, l=[], m=[])

        Now we instantiate the dataclass...
        >>> l = ListState(l=list("abc"), m=list("xyz"), meta="initial")
        >>> l  # doctest: +NORMALIZE_WHITESPACE
        ListState(meta='initial',
                  parent=None,
                  l=['a', 'b', 'c'],
                  m=['x', 'y', 'z'])

        ... and can add deltas to it. `l` will be extended:
        >>> l + Delta(l=list("def"), meta="first")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(meta='first',
                  parent=ListState...,
                  l=['a', 'b', 'c', 'd', 'e', 'f'],
                  m=['x', 'y', 'z'])

        ... wheras `m` will be replaced:
        >>> l + Delta(m=list("uvw"), meta="second")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(meta='second',
                  parent=ListState...,
                  l=['a', 'b', 'c'],
                  m=['u', 'v', 'w'])

        ... they can be chained:
        >>> l + Delta(l=list("d")) + Delta(m=list("u"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(...
                  l=['a', 'b', 'c', 'd'],
                  m=['u'])

        ... and we update multiple fields with one Delta:
        >>> l + Delta(l=list("ghi"), m=list("rst"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(...
                  l=['a', 'b', 'c', 'g', 'h', 'i'],
                  m=['r', 's', 't'])

        A non-existent field will be ignored.
        >>> l + Delta(o="not a field")  # doctest: +ELLIPSIS
        ListState(..., l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... but a warning will be emitted:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     _ = l + Delta(o="not a field")
        ...     print(w[0].message) # doctest: +NORMALIZE_WHITESPACE
        These fields: ['o'] could not be used to update ListState,
        which has these fields & aliases: ['meta', 'parent', 'l', 'm']

        We can also use the `.update` method instead of `__add__` to do the same thing:
        >>> l.update(l=list("ghi"), m=list("rst"))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        ListState(...,
                  l=['a', 'b', 'c', 'g', 'h', 'i'],
                  m=['r', 's', 't'])

        We can also define fields which `append` the last result:
        >>> @dataclass(frozen=True)
        ... class AppendState(StateHistory):
        ...    n: List = field(default_factory=list, metadata={"delta": "append"})

        >>> m = AppendState(n=list("ɑβɣ"))
        >>> m  # doctest: +ELLIPSIS
        AppendState(..., n=['ɑ', 'β', 'ɣ'])

        `n` will be appended:
        >>> m + Delta(n="∂")  # doctest: +ELLIPSIS
        AppendState(..., n=['ɑ', 'β', 'ɣ', '∂'])

        The metadata key "converter" is used to coerce types (inspired by
        [PEP 712](https://peps.python.org/pep-0712/)):
        >>> @dataclass(frozen=True)
        ... class CoerceStateList(StateHistory):
        ...    o: Optional[List] = field(default=None, metadata={"delta": "replace"})
        ...    p: List = field(default_factory=list, metadata={"delta": "replace",
        ...                                                    "converter": list})

        >>> r = CoerceStateList()

        If there is no `metadata["converter"]` set for a field, no coercion occurs
        >>> r + Delta(o="not a list")  # doctest: +ELLIPSIS
        CoerceStateList(..., o='not a list', p=[])

        If there is a `metadata["converter"]` set for a field, the data are coerced:
        >>> r + Delta(p="not a list")  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        CoerceStateList(...
                        o=None, p=['n', 'o', 't', ' ', 'a', ' ', 'l', 'i', 's', 't'])

        If the input data are of the correct type, they are returned unaltered:
        >>> r + Delta(p=["a", "list"])  # doctest: +ELLIPSIS
        CoerceStateList(..., o=None, p=['a', 'list'])

        With a converter, inputs are converted to the type output by the converter:
        >>> import pandas as pd
        >>> @dataclass(frozen=True)
        ... class CoerceStateDataFrame(StateHistory):
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




    """

    meta: Any = field(default=None, metadata={"delta": "replace"})

    def __add__(self, other):
        new = super().__add__(other, warn_on_unused_fields=True)
        _history.append(())
        new = replace(new, parent=self)
        return new

    @property
    def history(self):
        next_ancestor = self.parent
        while next_ancestor is not None:
            yield next_ancestor
            previous_ancestor = next_ancestor
            next_ancestor = previous_ancestor.parent