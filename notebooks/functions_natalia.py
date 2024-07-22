import pandas as pd
import numpy as np
import plotly.express as px


# Perform initial data cheking:
def initial_data_checking(df):
    # Print the shape of the DataFrame (number of rows and columns)
    print("\nShape of the DataFrame:\n")
    print(df.shape)

    # Print the count of duplicate rows
    print("\nDuplicate Rows Number:\n")
    print(df.duplicated().sum())

    # Print summary statistics for numerical columns
    print("\nSummary Statistics:\n")
    print(df.describe())


# Check Unique values, %, Missing values, %, data type:
def unique_and_missing_values_dtype(df):

    # Non-null counts and data types
    non_null_counts = df.notnull().sum()
    dtypes = df.dtypes

    # Count of unique values
    unique_count = df.nunique()

    # Percentage of unique values
    unique_percentage = (df.nunique() / len(df)) * 100

    # Count of missing values
    missing_count = df.isnull().sum()

    # Percentage of missing values
    missing_percentage = df.isnull().mean() * 100

    # Combine into a DataFrame
    summary = pd.DataFrame({
        'non-Null_count': non_null_counts,
        'dtype': dtypes,
        'unique_values': unique_count,
        '%_unique': unique_percentage.round(2).astype(str) + '%',
        'missing_values': missing_count,
        '%_missing': missing_percentage.round(2).astype(str) + '%'
    })

    return summary


# separate categorical and numerical columns:
def separate_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    print("\ncategorical_cols:\n")
    print(categorical_cols)
    print("\nnumerical_cols:\n")
    print(numerical_cols)

    # analyze_numerical cols:


def analyze_numerical(df: pd.DataFrame) -> pd.DataFrame:
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Perform descriptive analysis on numerical columns
    numerical_desc = df[numerical_cols].describe()

    # Display the resulting DataFrame
    print("\nNumerical Columns Analysis:")

    return numerical_desc


# analyze_categorical cols:
def analyze_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category'])

    # Perform descriptive analysis on categorical columns
    categorical_desc = categorical_cols.describe()

    return categorical_desc


# script for filling missing data (numerical_data with 'mean',categorical_data with 'mode' :

def fill_missing_values(df):

    # Select categorical and numerical columns
    categorical_data = df.select_dtypes(include=['object', 'category']).columns
    numerical_data = df.select_dtypes(include=['number']).columns

    # Fill missing values in numerical columns with the mean
    for column in numerical_data:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing values in categorical columns with the mode
    for column in categorical_data:
        if not df[column].mode().empty:  # Check if mode exists
            df[column] = df[column].fillna(df[column].mode()[0])

    return df
