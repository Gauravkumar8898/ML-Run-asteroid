import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.constant import transformed_dataset_path, columns_to_encode, plt_drop
import mlrun

df = pd.read_csv(transformed_dataset_path)


def load_dataset(file_path):
    """
    Loads the dataset from the specified file path using pandas.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        DataFrame: DataFrame containing the loaded dataset.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File not found at the specified path.")
    try:
        dataset = pd.read_csv(file_path)
        if dataset.empty:
            raise ValueError("File is empty.")
        return pd.DataFrame(dataset)

    except Exception as e:
        raise f"Error loading dataset: {str(e)}"


def drop_columns(data_frame, dropping_columns):
    """
    Drops specified columns from the given dataframe.

    Args:
        data_frame (DataFrame): DataFrame from which columns are to be dropped.
        dropping_columns (list): List of column names to be dropped.

    Returns:
        DataFrame: DataFrame after dropping specified columns.
    """
    try:
        validate_columns_exist(data_frame, dropping_columns)
        return data_frame.drop(dropping_columns, axis=1)

    except Exception as e:
        raise KeyError(f"Exception raised at drop_columns:str{e}")


def column_encode(df, columns):
    """
    Encodes the specified columns in the given DataFrame.

    Args:
        df (DataFrame): DataFrame to perform encoding on.
        columns (list): List of column names to encode.

    Returns:
        DataFrame: DataFrame with specified columns encoded.
    """
    try:
        for col in columns:
            df[col] = np.where(df[col] == True, 1, 0)
        return df

    except Exception as e:
        raise f"Error performing column encoding: {str(e)}"


def set_values_based_on_condition(df, column, condition_column, condition_value, first_value, second_value):
    """
    Set values in a column based on a condition using NumPy's np.where() function.

    Args:
        df (DataFrame): DataFrame containing the column to set values.
        column (str): Name of the column to set values.
        condition_column (str): Name of the column containing the condition.
        condition_value: Value to compare against in the condition column.
        first_value: Value to assign when condition is true.
        second_value: Value to assign when condition is false.

    Returns:
        DataFrame: DataFrame with the values set in the specified column based on the condition.
    """
    # Check if the DataFrame is empty
    if df.empty:
        return df

    try:
        validate_columns_exist(df, [condition_column])
        df[column] = np.where(df[condition_column] == condition_value, first_value, second_value)
        return df

    except Exception as e:
        raise f"Exception raised at validate_columns:{e}"


def validate_columns_exist(df, columns):
    """
    Validate if the specified columns exist in the DataFrame.

    Args:
        df (DataFrame): DataFrame to validate columns against.
        columns (list): List of column names to validate.

    Raises:
        KeyError: If any of the specified columns do not exist in the DataFrame.
    """
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise KeyError(f"Columns {', '.join(missing_columns)} do not exist in the DataFrame.")


def prepare_and_plot_ts_heatmap(df):
    """
    Prepares time series data and plots a heatmap of correlations.

    Args:
        df (DataFrame): DataFrame containing time series data.

    Returns:
        None
    """

    df = df.drop(plt_drop, axis=1)

    # Assuming your DataFrame has a 'Date' column as the index
    df.index = pd.to_datetime(df.index)

    # If your data is not already sorted by date, sort it
    df = df.sort_index()

    # Plotting heatmap of correlations
    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.savefig("/home/knoldus/PycharmProjects/asteroid_ml_wit/src/plot/heatmap.jpg")
    plt.show()


def plot_correlation_matrix(df):
    """
    Plots the correlation matrix heatmap for a DataFrame.

    Args:
        df (DataFrame): Input DataFrame.

    Returns:
        None
    """
    df = df.drop(plt_drop, axis=1)

    # Calculate the correlation matrix
    corr = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(20, 8))
    sns.heatmap(corr, annot=True, cmap='magma', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.savefig("/home/knoldus/PycharmProjects/asteroid_ml_wit/src/plot/corr_mat.jpg")
    plt.show()


def load_project(project_name):
    """
    Loading the project

    raise HttpStatusError for Mlrun if found
    :raise Exception if unknown exception occurs
    """
    try:
        return mlrun.get_or_create_project(f"{project_name}", context="./", user_project=True)
    except mlrun.errors.MLRunHTTPStatusError as error1:
        error1.strerror = 'Connection to port not found:'
        raise error1
