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


# (Nat)f_1.1.Perform initial data cheking:
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


# (Nat)f_1.2.Check Unique values, %, Missing values, %, data type:
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


# (Nat)f_1.3.separate categorical and numerical columns:
def separate_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    print("\ncategorical_cols:\n")
    print(categorical_cols)
    print("\nnumerical_cols:\n")
    print(numerical_cols)


# (Nat)f_1.4.Analyze_numerical cols
def analyze_numerical(df: pd.DataFrame) -> pd.DataFrame:
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Perform descriptive analysis on numerical columns
    numerical_desc = df[numerical_cols].describe()

    # Display the resulting DataFrame
    print("\nNumerical Columns Analysis:")

    return numerical_desc


# (Nat)f_1.5.analyze_categorical cols:
def analyze_categorical(df: pd.DataFrame) -> pd.DataFrame:
    # Select categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category'])

    # Perform descriptive analysis on categorical columns
    categorical_desc = categorical_cols.describe()

    return categorical_desc


# (Nat)f_1.6.script for filling missing data (numerical_data with 'mean',categorical_data with 'mode' :
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


# (Nat)f_1.7.Merging df_demo and df_experiment, df_web, on client_id:
def merge_3(df_demo, df_experiment, df_web):
    # Merges three dataframes on the 'client_id' column.
    # Merging df_demo and df_experiment on client_id
    merged_df = pd.merge(df_demo, df_experiment, on='client_id')

    # Merging the resulting dataframe with df_web on client_id
    final_merged_df = pd.merge(merged_df, df_web, on='client_id')

    return final_merged_df


# (Nat)f_2.1.Convert to datetime
def process_datetime(df, datetime_column='date_time', sort_columns=['visit_id', 'date_time']):
    # Convert to datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    # Sort
    df_sorted = df.sort_values(by=sort_columns)

    return df_sorted


# (Nat)f_2.2.Check that each visit_id has only 1 "confirm" step
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


# (Nat)f_2.3.Remove duplicated 'confirm' steps (keep only 1st occurrence)
def remove_duplicate_confirms(df):
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


"""Natalia's Functions related to completion_rate calculations"""


# (Nat)f_2.4.Calculate Completion Rate per day
def completion_rate_day(df):
    # Extract the date part from 'date_time'
    df['date'] = df['date_time'].dt.date

    # Group by date, variation, and 'visit_id', then check if 'confirm' is present for each group
    confirmations = df[df['process_step'] == 'confirm'].groupby(
        ['date', 'variation'])['visit_id'].nunique()

    # Count total unique 'visit_id' per day and variation
    total_visits = df.groupby(['date', 'variation'])['visit_id'].nunique()

    # Prepare the output DataFrame
    output_df = pd.DataFrame({
        'completions': confirmations,
        'total_visits': total_visits
    })

    # Calculate the completion rate
    output_df['completion_rate_per_day'] = output_df['completions'] / \
        output_df['total_visits']

    # Reset index to have 'date' and 'variation' as regular columns
    output_df.reset_index(inplace=True)

    return output_df


# (Nat)f_2.5.calculate_difference_in_average_completion_rates :
def calculate_difference_in_avg_completion_rates(df):
    # Group by 'variation' and calculate the mean of 'completion_rate_per_day'
    average_rates = df.groupby('variation')['completion_rate_per_day'].mean()

    # Convert the Series to a DataFrame for better presentation
    average_rates_df = average_rates.reset_index()
    average_rates_df.columns = ['Variation', 'average_completion_rate']

    # Calculate the difference between the completion rates for 'Control' and 'Test'
    if 'Control' in average_rates_df['Variation'].values and 'Test' in average_rates_df['Variation'].values:
        control_rate = average_rates_df.loc[average_rates_df['Variation']
                                            == 'Control', 'average_completion_rate'].values[0]
        test_rate = average_rates_df.loc[average_rates_df['Variation']
                                         == 'Test', 'average_completion_rate'].values[0]
        difference = test_rate - control_rate
    else:
        difference = None  # Returns None if either 'Control' or 'Test' group is not found

    return average_rates_df, difference


# (Nat)f_2.6.Check if the data is normally distributed with Histogram in completion rate per day
def histogram_with_density_plot(df, column):
    # Reduce the size of the DataFrame by sampling
    plt.figure(figsize=(10, 4))
    sns.histplot(x=df[column], bins=100, kde=True)  # Adjust the number of bins
    plt.title(f'Histogram with Density Plot for {column}')
    plt.show()


# (Nat)f_2.7.Remove outliers from data time_per_visit with Standard Deviation Method.
def remove_outliers_std(df, column, num_std=3):
    # Calculate the mean and standard deviation
    mean = df[column].mean()
    std_dev = df[column].std()

    # Define outliers as those beyond num_std standard deviations from the mean
    lower_bound = mean - (num_std * std_dev)
    upper_bound = mean + (num_std * std_dev)

    # Filter out outliers
    df_filtered = df[(df[column] > lower_bound) & (df[column] < upper_bound)]

    return df_filtered


# (Nat)f_2.8.Remove outliers from original data time_per_visit with Interquartile Range (IQR) Method
def remove_outliers_iqr(df, column):
    # Calculate Q1 and Q3
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define outliers as those below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    df_filtered = df[(df[column] > lower_bound) & (df[column] < upper_bound)]

    return df_filtered


# (Nat)f_2.9.Transform data with Box Cox method to make it normally distributed
def apply_boxcox_and_plot(df, column_name):
    # Check if all values are positive, as Box-Cox requires positive data
    if (df[column_name] <= 0).any():
        # Add a small constant to shift all values to be > 0 if any are non-positive
        df[column_name] += (df[column_name].min() * -1) + 1e-5

    # Applying Box-Cox transformation
    transformed_data, _ = boxcox(df[column_name])

    # Plotting the transformed distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(transformed_data, kde=True)
    plt.title(f'Box-Cox Transformed {column_name}')
    plt.xlabel(f'Transformed {column_name}')
    plt.ylabel('Frequency')
    plt.show()

    # Return the transformed data as a new column in the dataframe
    df[f'{column_name}_BoxCox'] = transformed_data
    return df


# (Nat)f_2.10.Apply normalization (to transformed boxcox) col:
def normalize_column(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' does not exist in the DataFrame")

    # Extract the column to be normalized
    data_to_normalize = df[[column_name]].copy()

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    normalized_data = scaler.fit_transform(data_to_normalize)

    # Add the normalized data back to the DataFrame using .loc to avoid SettingWithCopyWarning
    normalized_column_name = column_name + '_Normalized'
    df.loc[:, normalized_column_name] = normalized_data

    return df


"""Natalia's Functions related to Time spent per visit, per step"""


# (Nat)calculating the time spent for each visit in min
def calculate_time_spent_min(df):
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


# (Nat)calculating the time spent for each visit in seconds:
def calculate_time_per_visit_sec(df):
    # Dropping the 'process_step' column if it exists
    if 'process_step' in df.columns:
        df = df.drop(columns=['process_step'])

    # Grouping by 'visit_id' and calculating the time spent for each visit
    df['time_per_visit_in_sec'] = df.groupby('visit_id')['date_time'].transform(
        lambda x: (x.max() - x.min()).total_seconds())

    # Converting time difference to seconds and changing the data type to int64
    df['time_per_visit_in_sec'] = df['time_per_visit_in_sec'].astype('int64')

    # Dropping duplicate rows based on 'visit_id' while keeping the first occurrence
    df = df.drop_duplicates(subset='visit_id')

    # Dropping the 'date_time' column
    df = df.drop(columns=['date_time'])

    return df


# (Nat)Check if the time spent for each visit is normally distributed with Histogram and Boxplot to visually check for outliers
def histogram_distribution_boxplot_outliers(df, column):
    # Set up the figure size and subplot layout
    # Adjust the figure size for better visibility
    plt.figure(figsize=(20, 10))

    # Histogram with Density Plot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    # Adjust the number of bins and enable KDE
    sns.histplot(x=df[column], bins=100, kde=True)
    plt.title(f'Histogram with Density Plot for {column}')

    # Boxplot for outliers
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot for {column}')

    plt.tight_layout()  # Adjusts plot parameters for better layout
    plt.show()


# (Nat)Shapiro-Wilk test to check if data time_per_visit is normally distributed
def shapiro_wilk_test(df, column):
    # Ensure the DataFrame and the specified column exist
    if column in df.columns:
        # Extract the column data
        data = df[column]

        # Perform the Shapiro-Wilk test
        stat, p = stats.shapiro(data)

        # Display the results
        print(
            f'Shapiro-Wilk Test for {column}: Statistics={stat:.3f}, p={p:.3f}')
        if p > 0.05:
            print("The data appears to be normally distributed (fail to reject H0).")
        else:
            print("The data does not appear to be normally distributed (reject H0).")
    else:
        print(f"Column '{column}' not found in DataFrame.")



""" Tobias' Functions """

def calculate_avg_errors_per_visit(df):
    df_visits = df[['unique_session_id', 'process_step', 'date_time', 'variation']].sort_values(by=["unique_session_id", "date_time"])
    process_step_dict = {'start': 0, 'step_1': 1, 'step_2': 2, 'step_3': 3, 'confirm': 4}
    df_visits['process_step_number'] = df_visits['process_step'].map(process_step_dict)
    df_visits['previous_step_number'] = df_visits.groupby('unique_session_id')['process_step_number'].shift()
    df_visits['step_diff'] = df_visits['process_step_number'] - df_visits['previous_step_number']
    df_visits['step_back'] = df_visits['step_diff'].apply(lambda x: True if x < 0 else False)
    total_errors_control = df_visits.loc[(df_visits["variation"] == "Control")]['step_back'].sum()
    total_errors_test = df_visits.loc[(df_visits["variation"] == "Test")]['step_back'].sum()
    total_visits_control = df_visits.loc[(df_visits["variation"] == "Control")]["unique_session_id"].nunique()
    total_visits_test = df_visits.loc[(df_visits["variation"] == "Test")]["unique_session_id"].nunique()
    error_rate_control = total_errors_control / total_visits_control
    error_rate_test = total_errors_test / total_visits_test
    return error_rate_control, error_rate_test