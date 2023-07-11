from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator

from autora.state.delta import State
from autora.variable import VariableCollection


@dataclass(frozen=True)
class BasicAERState(State):
    variables: VariableCollection = field(metadata={"delta": "replace"})
    conditions: pd.Series = field(
        default_factory=pd.Series, metadata={"delta": "replace"}
    )
    experiment_data: pd.DataFrame = field(
        default_factory=pd.DataFrame, metadata={"delta": "extend"}
    )
    model: Optional[BaseEstimator] = field(default=None, metadata={"delta": "replace"})
