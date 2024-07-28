import pandas as pd
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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


""" Natalia functions """

# (Nat)Merging df_demo and df_experiment, df_web, on client_id


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
