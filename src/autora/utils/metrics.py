import numpy as np


def norms(arr: np.ndarray) -> np.ndarray:
    """
    Calculate the norms along the first axis
    Examples:
        >>> import pandas as pd
        >>> from autora.utils.transform import to_array

        Simple dataframe with one condition
        >>> df = pd.DataFrame({'x_0': [.2, 2, 3]})

        First transform:
        >>> as_array = to_array(df)
        >>> norms(as_array)
        array([0.2, 2. , 3. ])

        >>> df_two_dim = pd.DataFrame({'x_0': [0, 1, 3], 'x_1': [1, 0, 4]})
        >>> as_array = to_array(df_two_dim)
        >>> norms(as_array)
        array([1., 1., 5.])

        For nested dataframes
        >>> df_nested = pd.DataFrame({
        ...     'x_0': [[0, 0], [0, 1], [1, 0], [3, 4]]
        ... })
        >>> as_array = to_array(df_nested)
        >>> norms(as_array)
        array([0., 1., 1., 5.])

        ... and deeply nested
        >>> df_nested_deep = pd.DataFrame({
        ...     'x_0': [[[0, 0], [0, 1]], [[3, 0], [0, 4]]]
        ... })
        >>> as_array = to_array(df_nested_deep)
        >>> norms(as_array)
        array([1., 5.])

        ... no matter how many columns
        >>> df_nested_deep_multi_column = pd.DataFrame({
        ...     'x_0': [[[0, 0], [0, 4]], [[1, 0], [0, 0]]],
        ...     'x_1': [[[0, 3], [0, 0]], [[0, 0], [0, 0]]]
        ... })
        >>> as_array = to_array(df_nested_deep_multi_column)
        >>> norms(as_array)
        array([5., 1.])
    """
    return np.array([np.linalg.norm(np.ravel(row)) for row in arr])


def distances(arr_1: np.ndarray, arr_2: np.ndarray) -> np.ndarray:
    """
    Calculate the euclidian distance between two arrays no matter their dimension along the
    first axis
    Examples:
        >>> import pandas as pd
        >>> from autora.utils.transform import to_array

        Simple dataframe with one condition
        >>> df_1 = pd.DataFrame({'x_0': [0, 1, 2]})
        >>> df_2 = pd.DataFrame({'x_0': [1, 2, 3]})

        First transform:
        >>> as_array_1 = to_array(df_1)
        >>> as_array_2 = to_array(df_2)
        >>> distances(as_array_1, as_array_2)
        array([1., 1., 1.])

        >>> df_two_dim_1 = pd.DataFrame({'x_0': [0, 1, 3], 'x_1': [1, 0, 4]})
        >>> df_two_dim_2 = pd.DataFrame({'x_0': [0, 1, 3], 'x_1': [1, 1, 4]})
        >>> as_array_1 = to_array(df_two_dim_1)
        >>> as_array_2 = to_array(df_two_dim_2)
        >>> distances(as_array_1, as_array_2)
        array([0., 1., 0.])

        For nested dataframes
        >>> df_nested_1 = pd.DataFrame({
        ...     'x_0': [[0, 0], [0, 2], [0, 2], [0, 10], [4, 0]]
        ... })
        >>> df_nested_2 = pd.DataFrame({
        ...     'x_0': [[1, 0], [0, 0], [0, 5], [0, 6], [0, 3]]
        ... })
        >>> as_array_1 = to_array(df_nested_1)
        >>> as_array_2 = to_array(df_nested_2)
        >>> distances(as_array_1, as_array_2)
        array([1., 2., 3., 4., 5.])

        ... and deeply nested
        >>> df_nested_deep_1 = pd.DataFrame({
        ...     'x_0': [[[0, 0], [0, 1]], [[6, 0], [0, 10]]]
        ... })
        >>> df_nested_deep_2 = pd.DataFrame({
        ...     'x_0': [[[0, 0], [0, 1]], [[3, 0], [0, 6]]]
        ... })
        >>> as_array_1 = to_array(df_nested_deep_1)
        >>> as_array_2 = to_array(df_nested_deep_2)
        >>> distances(as_array_1, as_array_2)
        array([0., 5.])

        ... no matter how many columns
        >>> df_nested_deep_multi_column_1 = pd.DataFrame({
        ...     'x_0': [[[0, 0], [0, 4]], [[1, 0], [0, 0]]],
        ...     'x_1': [[[0, 3], [0, 0]], [[0, 0], [0, 0]]]
        ... })
        >>> df_nested_deep_multi_column_2 = pd.DataFrame({
        ...     'x_0': [[[0, 0], [0, 4]], [[1, 0], [0, 0]]],
        ...     'x_1': [[[0, 3], [0, 0]], [[0, 0], [0, 0]]]
        ... })
        >>> as_array_1 = to_array(df_nested_deep_multi_column_1)
        >>> as_array_2 = to_array(df_nested_deep_multi_column_2)
        >>> distances(as_array_1, as_array_2)
        array([0., 0.])

    """
    # Check that the two arrays have the same shape
    assert arr_1.shape == arr_2.shape, "Arrays must have the same shape"

    # For each row, calculate the squared distance
    return np.sqrt(
        np.array(
            [np.sum((np.ravel(a) - np.ravel(b)) ** 2) for a, b in zip(arr_1, arr_2)]
        )
    )
