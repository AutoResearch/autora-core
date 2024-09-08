from typing import Union

import numpy as np
import pandas as pd


def to_array(arr: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """
    Transforms a pandas data frame to a numpy array
    Args:
        arr: the pandas data frame

    Returns:
        a numpy array

    Examples:
        Same result as np.array(df) if rows of df are one dimensional:
        >>> df_one = pd.DataFrame({
        ...     'x_0': [1, 2, 3],
        ...     'x_1': [4, 5, 6],
        ...     'x_2': [7, 8, 9]})
        >>> np.array_equal(np.array(df_one), to_array(df_one))
        True

        If the rows contain lists ...
        >>> df_list = pd.DataFrame({
        ...     'x_0': [[0, 0], [1, 0], [2, 0]],
        ...     'x_1': [[0, 1], [1, 1], [2, 1]],
        ...     'x_2': [[0, 2], [1, 2], [2, 2]]
        ... })
        >>> array_transformed = to_array(df_list)
        >>> array_cast = np.array(df_list)

        the results are not equal:
        >>> np.array_equal(array_transformed, array_cast)
        False

        The cast array contains objects which are hard to work with:
        >>> array_cast
        array([[list([0, 0]), list([0, 1]), list([0, 2])],
               [list([1, 0]), list([1, 1]), list([1, 2])],
               [list([2, 0]), list([2, 1]), list([2, 2])]], dtype=object)

        The transformed array containst vectors (numbers):
        >>> array_transformed
        array([[[0, 0],
                [0, 1],
                [0, 2]],
        <BLANKLINE>
               [[1, 0],
                [1, 1],
                [1, 2]],
        <BLANKLINE>
               [[2, 0],
                [2, 1],
                [2, 2]]])

        ... the same is true for arrays:
        >>> df_array = pd.DataFrame({
        ...     'x_0': [np.array([0, 0]), np.array([1, 0]), np.array([2, 0])],
        ...     'x_1': [np.array([0, 1]), np.array([1, 1]), np.array([2, 1])],
        ...     'x_2': [np.array([0, 2]), np.array([1, 2]), np.array([2, 2])]
        ... })
        >>> array_transformed = to_array(df_array)
        >>> array_cast = np.array(df_list)

        the results are not equal:
        >>> np.array_equal(array_transformed, array_cast)
        False

        The cast array contains objects which are hard to work with:
        >>> array_cast
        array([[list([0, 0]), list([0, 1]), list([0, 2])],
               [list([1, 0]), list([1, 1]), list([1, 2])],
               [list([2, 0]), list([2, 1]), list([2, 2])]], dtype=object)

        The transformed array containst vectors (numbers):
        >>> array_transformed
        array([[[0, 0],
                [0, 1],
                [0, 2]],
        <BLANKLINE>
               [[1, 0],
                [1, 1],
                [1, 2]],
        <BLANKLINE>
               [[2, 0],
                [2, 1],
                [2, 2]]])

        # This also works with more nesting:
        >>> df_nested = pd.DataFrame({
        ...     'x_0': [[[0,0],[1,1]], [[0,0],[2,2]]],
        ...     'x_1': [[[1,1],[1,1]], [[1,1],[2,2]]]
        ... })
        >>> to_array(df_nested)
        array([[[[0, 0],
                 [1, 1]],
        <BLANKLINE>
                [[1, 1],
                 [1, 1]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[0, 0],
                 [2, 2]],
        <BLANKLINE>
                [[1, 1],
                 [2, 2]]]])

        When the inner lists don't have the same shape, an error is thrown and one can use
        a flattening version of this (ATTENTION: when using the flattening version,
        information about which entry belongs to which condition is lost):
    """
    if isinstance(arr, np.ndarray):
        return arr

    _lst = [list(row) for _, row in arr.iterrows()]
    return np.array(_lst)


def to_array_flatten(arr: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """
    Flattens elements in a pandas DataFrame to resolve shape inconsistencies.

    Args:
        df: A pandas DataFrame or Series with inconsistent element shapes.

    Returns:
        A numpy array where all elements are flattened.

    Example:
        >>> df_inconsistent = pd.DataFrame({
        ...     'x_0': [0, 2, 4],
        ...     'x_1': [[1, 1], [3, 3], [5, 5]]
        ... })
        >>> to_array_flatten(df_inconsistent)
        array([[0, 1, 1],
               [2, 3, 3],
               [4, 5, 5]])
    """
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(
        [
            np.concatenate(
                [np.ravel(x) if isinstance(x, (list, np.ndarray)) else [x] for x in row]
            )
            for _, row in arr.iterrows()
        ]
    )
