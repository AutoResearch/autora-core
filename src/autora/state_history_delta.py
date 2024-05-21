from dataclasses import dataclass, field, replace
from typing import List

from autora.state import Delta, State


@dataclass(frozen=True)
class StateHistory(State):
    """
    Base object for dataclasses which use the Delta mechanism.

    Examples:
        >>> from dataclasses import dataclass, field
        >>> from typing import List, Optional

        We define a dataclass where each field (which is going to be delta-ed) has additional
        metadata "delta" which describes its delta behaviour.
        >>> @dataclass(frozen=True)
        ... class ListState(StateHistory):
        ...    l: List = field(default_factory=list, metadata={"delta": "extend"})
        ...    m: List = field(default_factory=list, metadata={"delta": "replace"})

        Now we instantiate the dataclass...
        >>> l = ListState(l=list("abc"), m=list("xyz"))
        >>> l
        ListState(history=[], l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... and can add deltas to it. `l` will be extended:
        >>> l + Delta(l=list("def"))  # doctest: +NORMALIZE_WHITESPACE
        ListState(history=[{'l': ['d', 'e', 'f']}],
                  l=['a', 'b', 'c', 'd', 'e', 'f'],
                  m=['x', 'y', 'z'])

        ... wheras `m` will be replaced:
        >>> l + Delta(m=list("uvw"))
        ListState(history=[{'m': ['u', 'v', 'w']}], l=['a', 'b', 'c'], m=['u', 'v', 'w'])

        ... they can be chained:
        >>> l + Delta(l=list("def")) + Delta(m=list("uvw"))  # doctest: +NORMALIZE_WHITESPACE
        ListState(history=[{'l': ['d', 'e', 'f']}, {'m': ['u', 'v', 'w']}],
                  l=['a', 'b', 'c', 'd', 'e', 'f'],
                  m=['u', 'v', 'w'])

        ... and we update multiple fields with one Delta:
        >>> l + Delta(l=list("ghi"), m=list("rst"))  # doctest: +NORMALIZE_WHITESPACE
        ListState(history=[{'l': ['g', 'h', 'i'], 'm': ['r', 's', 't']}],
                  l=['a', 'b', 'c', 'g', 'h', 'i'],
                  m=['r', 's', 't'])

        A non-existent field will be ignored but included in the history:
        >>> l + Delta(o="not a field")
        ListState(history=[{'o': 'not a field'}], l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... but will trigger a warning:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     _ = l + Delta(o="not a field")
        ...     print(w[0].message) # doctest: +NORMALIZE_WHITESPACE
        These fields: ['o'] could not be used to update ListState,
        which has these fields & aliases: ['history', 'l', 'm']

        We can also use the `.update` method to do the same thing:
        >>> l.update(l=list("ghi"), m=list("rst"))  # doctest: +NORMALIZE_WHITESPACE
        ListState(history=[{'l': ['g', 'h', 'i'], 'm': ['r', 's', 't']}],
                  l=['a', 'b', 'c', 'g', 'h', 'i'],
                  m=['r', 's', 't'])

        We can also define fields which `append` the last result:
        >>> @dataclass(frozen=True)
        ... class AppendState(StateHistory):
        ...    n: List = field(default_factory=list, metadata={"delta": "append"})

        >>> m = AppendState(n=list("ɑβɣ"))
        >>> m
        AppendState(history=[], n=['ɑ', 'β', 'ɣ'])

        `n` will be appended:
        >>> m + Delta(n="∂")
        AppendState(history=[{'n': '∂'}], n=['ɑ', 'β', 'ɣ', '∂'])

        The metadata key "converter" is used to coerce types (inspired by
        [PEP 712](https://peps.python.org/pep-0712/)):
        >>> @dataclass(frozen=True)
        ... class CoerceStateList(StateHistory):
        ...    o: Optional[List] = field(default=None, metadata={"delta": "replace"})
        ...    p: List = field(default_factory=list, metadata={"delta": "replace",
        ...                                                    "converter": list})

        >>> r = CoerceStateList()

        If there is no `metadata["converter"]` set for a field, no coercion occurs
        >>> r + Delta(o="not a list")
        CoerceStateList(history=[{'o': 'not a list'}], o='not a list', p=[])

        If there is a `metadata["converter"]` set for a field, the data are coerced:
        >>> r + Delta(p="not a list")  # doctest: +NORMALIZE_WHITESPACE
        CoerceStateList(history=[{'p': 'not a list'}],
                        o=None, p=['n', 'o', 't', ' ', 'a', ' ', 'l', 'i', 's', 't'])

        If the input data are of the correct type, they are returned unaltered:
        >>> r + Delta(p=["a", "list"])
        CoerceStateList(history=[{'p': ['a', 'list']}], o=None, p=['a', 'list'])

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

        ... or an array:
        >>> import numpy as np
        >>> (s + Delta(q=np.linspace([1, 2], [10, 15], 3))).q
              0     1
        0   1.0   2.0
        1   5.5   8.5
        2  10.0  15.0

        ... or a dictionary:
        >>> (s + Delta(q={"a": [1,2,3], "b": [4,5,6]})).q
           a  b
        0  1  4
        1  2  5
        2  3  6

        ... or a list:
        >>> (s + Delta(q=[11, 12, 13])).q
            0
        0  11
        1  12
        2  13

        ... but not, for instance, a string:
        >>> (s + Delta(q="not compatible with pd.DataFrame")).q
        Traceback (most recent call last):
        ...
        ValueError: DataFrame constructor not properly called!

        Without a converter:
        >>> @dataclass(frozen=True)
        ... class CoerceStateDataFrameNoConverter(StateHistory):
        ...    r: pd.DataFrame = field(default_factory=pd.DataFrame, metadata={"delta": "replace"})

        ... there is no coercion – the object is passed unchanged
        >>> t = CoerceStateDataFrameNoConverter()
        >>> (t + Delta(r=np.linspace([1, 2], [10, 15], 3))).r
        array([[ 1. ,  2. ],
               [ 5.5,  8.5],
               [10. , 15. ]])


        A converter can cast from a DataFrame to a np.ndarray (with a single datatype),
        for instance:
        >>> @dataclass(frozen=True)
        ... class CoerceStateArray(State):
        ...    r: Optional[np.ndarray] = field(default=None,
        ...                            metadata={"delta": "replace",
        ...                                      "converter": np.asarray})

        Here we pass a dataframe, but expect a numpy array:
        >>> (CoerceStateArray() + Delta(r=pd.DataFrame([("a",1), ("b",2)], columns=list("xy")))).r
        array([['a', 1],
               ['b', 2]], dtype=object)

        We can define aliases which can transform between different potential field
        names.

    """

    history: List[Delta] = field(default_factory=list)

    def __add__(self, other):
        new = super().__add__(other)
        new = replace(new, history=self.history + [other])
        return new
