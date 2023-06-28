"""Classes to represent cycle state $S$ as $S_n = S_{0} + \sum_{i=1}^n \Delta S_{i}"""
from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class State:
    data: Optional[Union[pd.DataFrame, np.typing.ArrayLike]]


@dataclasses.dataclass(frozen=True)
class StateDelta(State):
    kind: Literal["extend", "replace"]

    def __radd__(self, other: Union[State, StateDelta]):
        updates = dict()
        for f in dataclasses.fields(other):
            if (value := getattr(self, f.name)) is not None:
                if self.kind == "replace":
                    updates[f.name] = value
                elif self.kind == "extend":
                    other_value = getattr(other, f.name)
                    if isinstance(other_value, pd.DataFrame):
                        updates[f.name] = pd.concat(
                            (other_value, value), ignore_index=True
                        )
                    elif isinstance(other_value, np.ndarray):
                        updates[f.name] = np.row_stack([other_value, value])
        return dataclasses.replace(other, **updates)
