from typing import List

import pandas as pd

from autora.variable import IV


def align_dataframe_to_ivs(
    dataframe: pd.DataFrame, independent_variables: List[IV]
) -> pd.DataFrame:
    """
    Aligns a dataframe to a metadata object, ensuring that the columns are in the same order
    as the independent variables in the metadata.

    Args:
        dataframe: a dataframe with columns to align
        independent_variables: a list of independent variables

    Returns:
        a dataframe with columns in the same order as the independent variables in the metadata

    Examples:
        >>> df = pd.DataFrame({'y': [1,2,3], 'x': [4, 6, 6]})
        >>> v_names = [IV('x'), IV('y')]
        >>> align_dataframe_to_ivs(df, v_names)
           x  y
        0  4  1
        1  6  2
        2  6  3

        >>> align_dataframe_to_ivs(pd.DataFrame({'x': [1, 2, 3]}), v_names)
        Traceback (most recent call last):
        ...
        Exception: Independent variables ['x', 'y'] and columns of dataframe do not align\
 Index(['x'], dtype='object')



    """
    variable_names = list()
    if set(dataframe.columns) != set([v.name for v in independent_variables]):
        raise Exception(f'Independent variables {[v.name for v in independent_variables]} '
                        f'and columns of dataframe do not align {dataframe.columns}')
    for variable in independent_variables:
        variable_names.append(variable.name)
    return dataframe[variable_names]
