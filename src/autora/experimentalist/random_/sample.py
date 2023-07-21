from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.state.delta import Result, State, wrap_to_use_state


def _state(s: State, **kwargs) -> State:
    """
    Take a random sample from some input conditions.

    Args:
        s: a State object with a `variables` field.

    Returns: a State object updated with the new conditions

    Examples:
        >>> from autora.state.bundled import StandardState
        >>> s = StandardState(conditions=pd.DataFrame({"x": range(100, 200)}))
        >>> _state(s, random_state=1, replace=False, num_samples=3).conditions
              x
        80  180
        84  184
        33  133

    """
    return wrap_to_use_state(_result)(s, **kwargs)


def _result(
    conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
    num_samples: int = 1,
    random_state: Optional[int] = None,
    replace: bool = False,
) -> Result:
    """
    Take a random sample from some input conditions.

    Args:
        conditions: the conditions to sample from
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: a Result object with a field `conditions` containing a DataFrame of the sampled
    conditions

    Examples:
        From a pd.DataFrame:
        >>> import pandas as pd
        >>> _result(
        ...     pd.DataFrame({"x": range(100, 200)}), num_samples=5, random_state=180)["conditions"]
              x
        67  167
        71  171
        64  164
        63  163
        96  196

    """
    return Result(
        conditions=_base(
            conditions=conditions,
            num_samples=num_samples,
            random_state=random_state,
            replace=replace,
        )
    )


def _base(
    conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
    num_samples: int = 1,
    random_state: Optional[int] = None,
    replace: bool = False,
) -> pd.DataFrame:
    """
    Take a random sample from some input conditions.

    Args:
        conditions: the conditions to sample from
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: a Result object with a field `conditions` containing a DataFrame of the sampled
    conditions

    Examples:
        From a pd.DataFrame:
        >>> import pandas as pd
        >>> _base(
        ...     pd.DataFrame({"x": range(100, 200)}), num_samples=5, random_state=180)
              x
        67  167
        71  171
        64  164
        63  163
        96  196

    """
    return pd.DataFrame.sample(
        conditions, random_state=random_state, n=num_samples, replace=replace
    )
