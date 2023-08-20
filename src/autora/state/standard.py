from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator

from autora.state import State
from autora.variable import VariableCollection


@dataclass(frozen=True)
class StandardState(State):
    """
    Examples:
        The state can be initialized emtpy
        >>> from autora.state.delta import Delta
        >>> from autora.variable import VariableCollection, Variable
        >>> s = StandardState()
        >>> s
        StandardState(variables=None, conditions=None, experiment_data=None, models=[])

        The `variables` can be updated using a `Delta`:
        >>> dv1 = Delta(variables=VariableCollection(independent_variables=[Variable("1")]))
        >>> s + dv1
        StandardState(variables=VariableCollection(independent_variables=[Variable(name='1',...)

        ... and are replaced by each `Delta`:
        >>> dv2 = Delta(variables=VariableCollection(independent_variables=[Variable("2")]))
        >>> s + dv1 + dv2
        StandardState(variables=VariableCollection(independent_variables=[Variable(name='2',...)

        The `conditions` can be updated using a `Delta`:
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
        >>> s + Delta(conditions="not compatible with pd.DataFrame")
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
        >>> s + Delta(experiment_data="not compatible with pd.DataFrame")
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

        The last model is available under the `model` property:
        >>> (s + dm1 + dm2).model
        DummyClassifier(constant=3)

        If there is no model, `None` is returned:
        >>> print(s.model)
        None

        `models` can also be updated using a Delta with a single `model`:
        >>> dm3 = Delta(model=DummyClassifier(constant=4))
        >>> (s + dm1 + dm3).model
        DummyClassifier(constant=4)

        As before, the `models` list is extended:
        >>> (s + dm1 + dm3).models
        [DummyClassifier(constant=1), DummyClassifier(constant=4)]

        No coercion or validation occurs with `models` or `model`:
        >>> (s + dm1 + Delta(model="not a model")).models
        [DummyClassifier(constant=1), 'not a model']


    """

    variables: Optional[VariableCollection] = field(
        default=None, metadata={"delta": "replace"}
    )
    conditions: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "replace", "converter": pd.DataFrame}
    )
    experiment_data: Optional[pd.DataFrame] = field(
        default=None, metadata={"delta": "extend", "converter": pd.DataFrame}
    )
    models: List[BaseEstimator] = field(
        default_factory=list,
        metadata={"delta": "extend", "aliases": {"model": lambda model: [model]}},
    )

    @property
    def model(self):
        """Alias for the last model in the `models`."""
        try:
            return self.models[-1]
        except IndexError:
            return None
