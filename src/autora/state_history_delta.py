from dataclasses import dataclass, field, fields, replace
from typing import List

from autora.state import Delta, StandardState, State


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


@dataclass(frozen=True)
class StandardStateHistory(StandardState, StateHistory):
    """
    Examples:
        The state can be initialized emtpy
        >>> from autora.variable import VariableCollection, Variable
        >>> s = StandardStateHistory()
        >>> s  # doctest: +NORMALIZE_WHITESPACE
        StandardStateHistory(history=[{'variables': None,
                                       'conditions': None,
                                       'experiment_data': None,
                                       'models': []}],
                            variables=None, conditions=None, experiment_data=None, models=[])

        The `variables` can be updated using a `Delta`:
        >>> dv1 = Delta(variables=VariableCollection(independent_variables=[Variable("1")]))
        >>> s + dv1 # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StandardStateHistory(history=[...],
            variables=VariableCollection(independent_variables=[Variable(name='1', ...)], ...),
            ...)

        ... and are replaced by each `Delta`:
        >>> dv2 = Delta(variables=VariableCollection(independent_variables=[Variable("2")]))
        >>> s + dv1 + dv2 # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        StandardStateHistory(history=[...],
            variables=VariableCollection(independent_variables=[Variable(name='2',...)

        The `conditions` can be updated using a `Delta`:
        >>> import pandas as pd
        >>> dc1 = Delta(conditions=pd.DataFrame({"x": [1, 2, 3]}))
        >>> (s + dc1).conditions
           x
        0  1
        1  2
        2  3

        ... and are replaced by each `Delta`:
        >>> dc2 = Delta(conditions=pd.DataFrame({"x": [4, 5]}))
        >>> (s + dc1 + dc2).conditions
           x
        0  4
        1  5

        Datatypes other than `pd.DataFrame` will be coerced into a `DataFrame` if possible.
        >>> import numpy as np
        >>> dc3 = Delta(conditions=np.core.records.fromrecords([(8, "h"), (9, "i")], names="n,c"))
        >>> (s + dc3).conditions
           n  c
        0  8  h
        1  9  i

        If they are passed without column names, no column names are inferred.
        This is to ensure that accidental mislabeling of columns cannot occur.
        Column names should usually be provided.
        >>> dc4 = Delta(conditions=[(6,), (7,)])
        >>> (s + dc4).conditions
           0
        0  6
        1  7

        Datatypes which are incompatible with a pd.DataFrame will throw an error:
        >>> s + Delta(conditions="not compatible with pd.DataFrame") \
# doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...

        Experiment data can be updated using a Delta:
        >>> ded1 = Delta(experiment_data=pd.DataFrame({"x": [1,2,3], "y": ["a", "b", "c"]}))
        >>> (s + ded1).experiment_data
           x  y
        0  1  a
        1  2  b
        2  3  c

        ... and are extended with each Delta:
        >>> ded2 = Delta(experiment_data=pd.DataFrame({"x": [4, 5, 6], "y": ["d", "e", "f"]}))
        >>> (s + ded1 + ded2).experiment_data
           x  y
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        4  5  e
        5  6  f

        If they are passed without column names, no column names are inferred.
        This is to ensure that accidental mislabeling of columns cannot occur.
        >>> ded3 = Delta(experiment_data=pd.DataFrame([(7, "g"), (8, "h")]))
        >>> (s + ded3).experiment_data
           0  1
        0  7  g
        1  8  h

        If there are already data present, the column names must match.
        >>> (s + ded2 + ded3).experiment_data
             x    y    0    1
        0  4.0    d  NaN  NaN
        1  5.0    e  NaN  NaN
        2  6.0    f  NaN  NaN
        3  NaN  NaN  7.0    g
        4  NaN  NaN  8.0    h

        `experiment_data` other than `pd.DataFrame` will be coerced into a `DataFrame` if possible.
        >>> import numpy as np
        >>> ded4 = Delta(
        ...     experiment_data=np.core.records.fromrecords([(1, "a"), (2, "b")], names=["x", "y"]))
        >>> (s + ded4).experiment_data
           x  y
        0  1  a
        1  2  b

        `experiment_data` which are incompatible with a pd.DataFrame will throw an error:
        >>> s + Delta(experiment_data="not compatible with pd.DataFrame") \
# doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...

        `models` can be updated using a Delta:
        >>> from sklearn.dummy import DummyClassifier
        >>> dm1 = Delta(models=[DummyClassifier(constant=1)])
        >>> dm2 = Delta(models=[DummyClassifier(constant=2), DummyClassifier(constant=3)])
        >>> (s + dm1).models
        [DummyClassifier(constant=1)]

        >>> (s + dm1 + dm2).models
        [DummyClassifier(constant=1), DummyClassifier(constant=2), DummyClassifier(constant=3)]

    """
