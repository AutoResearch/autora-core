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
    """
    variable_names = list()
    for variable in independent_variables:
        variable_names.append(variable.name)
    return dataframe[variable_names]
