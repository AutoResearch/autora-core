import pandas as pd

from autora.variable import VariableCollection


def align_dataframe_to_metadata(
    dataframe: pd.DataFrame, metadata: VariableCollection
) -> pd.DataFrame:
    """
    Aligns a dataframe to a metadata object, ensuring that the columns are in the same order
    as the independent variables in the metadata.

    Args:
        dataframe: a dataframe with columns to align
        metadata: a VariableCollection with independent variables

    Returns:
        a dataframe with columns in the same order as the independent variables in the metadata
    """
    variable_names = list()
    for variable in metadata.independent_variables:
        variable_names.append(variable.name)
    return dataframe[variable_names]
