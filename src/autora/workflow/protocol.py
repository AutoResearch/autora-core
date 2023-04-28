from enum import Enum
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    TypeVar,
    Union,
    runtime_checkable,
)

from autora.variable import VariableCollection
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

State = TypeVar("State")


class ResultKind(str, Enum):
    """
    Kinds of results which can be held in the Result object.

    Examples:
        >>> ResultKind.CONDITION is ResultKind.CONDITION
        True

        >>> ResultKind.CONDITION is ResultKind.VARIABLES
        False

        >>> ResultKind.CONDITION == "CONDITION"
        True

        >>> ResultKind.CONDITION == "VARIABLES"
        False

        >>> ResultKind.CONDITION in {ResultKind.CONDITION, ResultKind.PARAMS}
        True

        >>> ResultKind.VARIABLES in {ResultKind.CONDITION, ResultKind.PARAMS}
        False
    """

    CONDITION = "CONDITION"
    OBSERVATION = "OBSERVATION"
    MODEL = "MODEL"
    PARAMS = "PARAMS"
    VARIABLES = "VARIABLES"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"


class SupportsDataKind(Protocol):
    """Object with attributes for `data` and `kind`"""

    data: Optional[Any]
    kind: Optional[ResultKind]


class SupportsControllerStateFields(Protocol):
    """Support representing snapshots of a controller state as mutable fields."""

    variables: VariableCollection
    params: Dict
    conditions: Sequence[ArrayLike]
    observations: Sequence[ArrayLike]
    models: Sequence[BaseEstimator]

    def update(self: State, **kwargs) -> State:
        ...


class SupportsControllerStateProperties(Protocol):
    """Support representing snapshots of a controller state as immutable properties."""

    def update(self: State, **kwargs) -> State:
        ...

    @property
    def variables(self) -> VariableCollection:
        ...

    @property
    def params(self) -> Dict:
        ...

    @property
    def conditions(self) -> Sequence[ArrayLike]:
        ...

    @property
    def observations(self) -> Sequence[ArrayLike]:
        ...

    @property
    def models(self) -> Sequence[BaseEstimator]:
        ...


SupportsControllerState = Union[
    SupportsControllerStateFields, SupportsControllerStateProperties
]


class SupportsControllerStateHistory(SupportsControllerStateProperties, Protocol):
    """Represents controller state as a linear sequence of entries."""

    def __init__(self, history: Sequence[SupportsDataKind]):
        ...

    def filter_by(self: State, kind: Optional[Set[Union[str, ResultKind]]]) -> State:
        ...

    @property
    def history(self) -> Sequence[SupportsDataKind]:
        ...


class Executor(Protocol):
    """A Callable which, given some state, and some parameters, returns an updated state."""

    def __call__(self, __state: State, __params: Dict) -> State:
        ...


ExecutorCollection = Mapping[str, Executor]


@runtime_checkable
class SupportsLoadDump(Protocol):
    def dump(self, data, file) -> None:
        ...

    def load(self, file) -> Any:
        ...
