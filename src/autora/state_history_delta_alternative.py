from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator

from autora.state_history_delta import DeltaHistory
from autora.variable import VariableCollection


@dataclass(frozen=True)
class AlternateDeltaHistory(DeltaHistory):
    """
    Examples:
        The state can be initialized emtpy
        >>> from autora.variable import VariableCollection, Variable
        >>> from autora.state import Delta
        >>> s = AlternateDeltaHistory()
        >>> s  # doctest: +NORMALIZE_WHITESPACE
        AlternateDeltaHistory(history=[{'variables': None,
                                       'conditions': None,
                                       'experiment_data': None,
                                       'model': None}],
                            variables=None, conditions=None, experiment_data=None, model=None)

        The `variables` can be updated using a `Delta`:
        >>> dv1 = Delta(variables=VariableCollection(independent_variables=[Variable("1")]))
        >>> s + dv1 # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        AlternateDeltaHistory(history=[...],
            variables=VariableCollection(independent_variables=[Variable(name='1', ...)], ...),
            ...)

        ... and are replaced by each `Delta`:
        >>> dv2 = Delta(variables=VariableCollection(independent_variables=[Variable("2")]))
        >>> s + dv1 + dv2 # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        AlternateDeltaHistory(history=[...],
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
        >>> ded1 = Delta(experiment_data=pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}))
        >>> (s + ded1).experiment_data
           x  y
        0  1  a
        1  2  b
        2  3  c

        ... and are replaced with each Delta:
        >>> ded2 = Delta(experiment_data=pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})\
                    .mask(lambda d: d.y == "b"))
        >>> (s + ded1 + ded2).experiment_data
             x    y
        0  1.0    a
        1  NaN  NaN
        2  3.0    c

        If they are passed without column names, no column names are inferred.
        This is to ensure that accidental mislabeling of columns cannot occur.
        >>> ded3 = Delta(experiment_data=pd.DataFrame([(7, "g"), (8, "h")]))
        >>> (s + ded3).experiment_data
           0  1
        0  7  g
        1  8  h


        `experiment_data` other than `pd.DataFrame` will be coerced into a `DataFrame` if possible.
        >>> import numpy as np
        >>> ded4 = Delta(
        ...     experiment_data=np.core.records.fromrecords([(1, "a"), (2, "b")], names=["x", "y"]))
        >>> (s + ded4).experiment_data
           x  y
        0  1  a
        1  2  b

        `experiment_data` which are incompatible with a pd.DataFrame will throw an error:
        >>> ded5 = Delta(experiment_data="not compatible with pd.DataFrame")
        >>> s + ded5 # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: ...

        `models` can be updated using a Delta:
        >>> from sklearn.dummy import DummyClassifier
        >>> dm1 = Delta(model=DummyClassifier(constant=1))
        >>> dm2 = Delta(model=DummyClassifier(constant=2))
        >>> dm3 = Delta(model=DummyClassifier(constant=3))
        >>> (s + dm1).model
        DummyClassifier(constant=1)

        >>> (s + dm1 + dm2 + dm3).model
        DummyClassifier(constant=3)


        The history can be accessed to get older variants of any of the field versions:
        >>> sh = (s + dm1 + dm2 + dm3)
        >>> sh.history_of("model") # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [None,
         DummyClassifier(constant=1),
         DummyClassifier(constant=2),
         DummyClassifier(constant=3)]

        For more involved filtering, potentially involving other metadata, the .history_filter
        method can be used:
        >>> dme1 = Delta(model=DummyClassifier(constant=1))
        >>> dme2 = Delta(model=DummyClassifier(constant=2), meta="flag")
        >>> dme3 = Delta(model=DummyClassifier(constant=3), conditions=[1,2,3]) # multi-update
        >>> shm = (s + dme1 + dme2 + dme3)

        Filter the history for deltas containing a field called "meta" with the value "flag"
        >>> shm.history_filter(lambda s: s.get("meta") == "flag")
        [{'model': DummyClassifier(constant=2), 'meta': 'flag'}]

        You can do the same thing by filtering the history directly
        >>> list(filter(lambda s: s.get("meta") == "flag", shm.history))
        [{'model': DummyClassifier(constant=2), 'meta': 'flag'}]

        Filter the history for deltas containing both a model and conditions
        >>> shm.history_filter(
        ...     lambda s: ("model" in s and "conditions" in s)) # doctest: +NORMALIZE_WHITESPACE
        [{'variables': None, 'conditions': None, 'experiment_data': None, 'model': None},
         {'model': DummyClassifier(constant=3), 'conditions': [1, 2, 3]}]

        ... or extract just the models from those entries:
        >>> [m["model"]
        ...     for m in filter(lambda s: ("model" in s and "conditions" in s), shm.history)
        ... ] # doctest: +NORMALIZE_WHITESPACE
        [None, DummyClassifier(constant=3)]

    """

    variables: Optional[VariableCollection] = field(
        default=None, metadata={"delta": "replace"}
    )
    conditions: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "replace", "converter": pd.DataFrame}
    )
    experiment_data: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "replace", "converter": pd.DataFrame}
    )
    model: BaseEstimator = field(
        default=None,
        metadata={"delta": "replace"},
    )
