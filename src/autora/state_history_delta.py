from dataclasses import dataclass, field, fields, replace
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

        If we instantiate the object with no data, the data fields will be initialized with
        default values and the history will be initialized with a dictionary of those default
        values.
        >>> ListState()
        ListState(history=[{'l': [], 'm': []}], l=[], m=[])

        If instead we specify that the history be a list containing `None`, that value is used to
        initialize the history.
        >>> ListState(history=[None], l=[1], m=[2])
        ListState(history=[None], l=[1], m=[2])

        If the history is initialized as an empty list (the default), then the history is
        initialized with the initial values of the other fields.
        >>> ListState(history=[], l=[1], m=[2])
        ListState(history=[{'l': [1], 'm': [2]}], l=[1], m=[2])

        Now we instantiate the dataclass...
        >>> l = ListState(l=list("abc"), m=list("xyz"))
        >>> l  # doctest: +NORMALIZE_WHITESPACE
        ListState(history=[{'l': ['a', 'b', 'c'], 'm': ['x', 'y', 'z']}],
                  l=['a', 'b', 'c'],
                  m=['x', 'y', 'z'])

        ... and can add deltas to it. `l` will be extended:
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
        ... class AppendState(StateHistory):
        ...    n: List = field(default_factory=list, metadata={"delta": "append"})

        >>> m = AppendState(n=list("ɑβɣ"))
        >>> m
        AppendState(history=[{'n': ['ɑ', 'β', 'ɣ']}], n=['ɑ', 'β', 'ɣ'])

        `n` will be appended:
        >>> m + Delta(n="∂")  # doctest: +ELLIPSIS
        AppendState(history=[..., {'n': '∂'}], n=['ɑ', 'β', 'ɣ', '∂'])

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
        CoerceStateList(history=[{'o': None, 'p': []}, {'o': 'not a list'}], o='not a list', p=[])

        If there is a `metadata["converter"]` set for a field, the data are coerced:
        >>> r + Delta(p="not a list")  # doctest: +NORMALIZE_WHITESPACE
        CoerceStateList(history=[{'o': None, 'p': []}, {'p': 'not a list'}],
                        o=None, p=['n', 'o', 't', ' ', 'a', ' ', 'l', 'i', 's', 't'])

        If the input data are of the correct type, they are returned unaltered:
        >>> r + Delta(p=["a", "list"])  # doctest: +ELLIPSIS
        CoerceStateList(history=[..., {'p': ['a', 'list']}], o=None, p=['a', 'list'])

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
            initial_data_except_history = {
                f.name: self.__getattribute__(f.name)
                for f in fields(self)
                if f.name != "history"
            }
            self.history.append(initial_data_except_history)
