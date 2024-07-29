import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets, model_selection, metrics
from scipy.stats import ttest_ind
from scipy import stats
from scipy.stats import boxcox


# Data exploration

def data_exploration(df):

    # check number of rows and columns
    shape = df.shape
    print("Number of rows:", shape[0])
    print("Number of columns:", shape[1])

    # check duplicates
    check_duplicates = df.duplicated().sum()
    print("Number of duplicates:", check_duplicates)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })

    # Reset index to make 'Column' a regular column
    summary_df.reset_index(drop=True, inplace=True)

    # Display the summary DataFrame
    summary_df

    # check numerical columns
    numerical_columns = df.select_dtypes("number").columns
    print("\nNumerical Columns:", numerical_columns)

    # check categorical columns
    categorical_columns = df.select_dtypes("object").columns
    print("\nCategorical Columns:", categorical_columns)

    return summary_df


def rename_demo_columns(df):

    new_column_names = {
        "clnt_tenure_yr": "tenure_year",
        "clnt_tenure_mnth": "tenure_month",
        "clnt_age": "age",
        "gendr": "gender",
        "num_accts": "number_of_accounts",
        "bal": "balance",
        "calls_6_mnth": "calls_6_month",
        "logons_6_mnth": "logons_6_month"
    }
    df_renamed = df.rename(columns=new_column_names).copy()

    return df_renamed


def fill_missing(df, cols_numerical, cols_categorical):

    # Fill missing values in numerical columns with the mean
    for column in cols_numerical:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing values in categorical columns with the mode
    for column in cols_categorical:
        if not df[column].mode().empty:  # Check if mode exists
            df[column] = df[column].fillna(df[column].mode()[0])

    df_filled = df.copy()

    return df_filled


def plot_distributions_numerical(df, cols):
    height = 2*len(cols)
    fig, axs = plt.subplots(len(cols), 1, figsize=(5, height))
    k = 0

    if len(cols) > 1:
        for col in cols:
            if df.dtypes.astype(str)[col] == 'float64' or df.dtypes.astype(str)[col] == 'int64':
                sns.histplot(data=df, x=col, ax=axs[k])
            elif df.dtypes.astype(str)[col] == 'object':
                sns.countplot(data=df, x=col, ax=axs[k])
            k += 1
    else:
        col = cols[0]
        if df.dtypes.astype(str)[col] == 'float64' or df.dtypes.astype(str)[col] == 'int64':
            sns.histplot(data=df, x=col, ax=axs)
        elif df.dtypes.astype(str)[col] == 'object':
            sns.countplot(data=df, x=col, ax=axs)

    fig.tight_layout()


def plot_distributions_categorical(df, cols):
    height = 2*len(cols)
    fig, axs = plt.subplots(len(cols), 1, figsize=(5, height))
    k = 0
    for col in cols:
        sns.countplot(data=df, x=col, ax=axs[k])
        k += 1
    fig.tight_layout()




""" Natalia's functions """
# (Nat)Perform initial data cheking:
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


# (Nat)Check Unique values, %, Missing values, %, data type:
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


# (Nat)separate categorical and numerical columns:
def separate_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    print("\ncategorical_cols:\n")
    print(categorical_cols)
    print("\nnumerical_cols:\n")
    print(numerical_cols)


# (Nat) Analyze_numerical cols
def analyze_numerical(df: pd.DataFrame) -> pd.DataFrame:
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Perform descriptive analysis on numerical columns
    numerical_desc = df[numerical_cols].describe()

    # Display the resulting DataFrame
    print("\nNumerical Columns Analysis:")

    return numerical_desc


# (Nat)analyze_categorical cols:
def analyze_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category'])

    # Perform descriptive analysis on categorical columns
    categorical_desc = categorical_cols.describe()

    return categorical_desc


# (Nat)script for filling missing data (numerical_data with 'mean',categorical_data with 'mode' :
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


# (Nat)Merging df_demo and df_experiment, df_web, on client_id:
def merge_3(df_demo, df_experiment, df_web):
    # Merges three dataframes on the 'client_id' column.
    # Merging df_demo and df_experiment on client_id
    merged_df = pd.merge(df_demo, df_experiment, on='client_id')

    # Merging the resulting dataframe with df_web on client_id
    final_merged_df = pd.merge(merged_df, df_web, on='client_id')

    return final_merged_df


# (Nat)Convert to datetime
def process_datetime(df, datetime_column='date_time', sort_columns=['visit_id', 'date_time']):
    # Convert to datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    # Sort
    df_sorted = df.sort_values(by=sort_columns)

    return df_sorted


# (Nat)Check that each visit_id has only 1 "confirm" step
def count_duplicate_confirms(df):
    """
    Count and return the number of visit_ids with more than one 'confirm' step in the given DataFrame.
    """
    # Count the number of 'confirm' steps for each visit_id
    confirm_counts = df[df['process_step'] ==
                        'confirm'].groupby('visit_id').size()
    # Filter visit_id that have more than one 'confirm' step
    duplicate_confirms = confirm_counts[confirm_counts > 1]
    # Return the number of duplicates
    print(f"duplicated 'confirm' steps: {len(duplicate_confirms)}")


# (Nat)Remove duplicated 'confirm' steps (keep only 1st occurrence)
def remove_duplicate_confirms(df):
    """
    Remove duplicated 'confirm' steps in the DataFrame, keeping only the first occurrence,
    and reset the index of the resulting DataFrame.
    """
    # Identify rows with 'confirm' steps
    confirm_rows = df[df['process_step'] == 'confirm']

    # Identify duplicated 'confirm' steps, keeping only the first occurrence
    duplicated_confirms = confirm_rows.duplicated(
        subset=['visit_id', 'process_step'], keep='first')

    # Get indices of the duplicated 'confirm' steps
    indices_to_remove = confirm_rows[duplicated_confirms].index

    # Remove the duplicated 'confirm' steps from the original DataFrame
    df_cleaned = df.drop(indices_to_remove).reset_index(drop=True)

    return df_cleaned


# (Nat)calculating the time spent for each visit
def calculate_time_spent(df):
    # Dropping the 'process_step' column
    df = df.drop(columns=['process_step'])

    # Grouping by 'visit_id' and calculating the time spent for each visit
    df['total_time_per_visit_id'] = df.groupby(
        'visit_id')['date_time'].transform(lambda x: x.max() - x.min())

    # Dropping duplicate rows based on 'visit_id' while keeping the first occurrence
    df = df.drop_duplicates(subset='visit_id')

    # Dropping the 'date_time' column
    df = df.drop(columns=['date_time'])

    return df
