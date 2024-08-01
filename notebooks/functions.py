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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import scipy.stats as st

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


"""Functions for Completion Rate analysis"""


def main_df_hypothesis_1(df1, df2, df3, column):

    # merge all the dataframes
    df_merge_1 = pd.merge(df1, df2, on=column, how='inner')
    df_merge_2 = pd.merge(df_merge_1, df3, on=column, how='inner')
    
    # format datetime and add date and month column
    df_merge_2['date_time'] = pd.to_datetime(df_merge_2['date_time'], errors='coerce')
    df_merge_2['date'] = df_merge_2['date_time'].dt.date
    df_merge_2['date'] = pd.to_datetime(df_merge_2['date'], errors='coerce')
    df_merge_2['month'] = df_merge_2['date_time'].dt.strftime('%B')
    
    # drop irrelevent column
    irrelevant_columns = ['tenure_month', 'balance', 'number_of_accounts', 'calls_6_month', 'logons_6_month', 'date_time']
    df_merge_2 = df_merge_2.drop(columns=irrelevant_columns).drop_duplicates().reset_index(drop=True)

    # categorize process_step
    df_merge_2['step_check'] = df_merge_2.groupby('visit_id')['process_step'].transform(lambda x: 'confirm' if 'confirm' in x.values else 'no_confirm')
    df_merge_2 = df_merge_2.drop(columns='process_step').drop_duplicates().reset_index(drop=True)

    return df_merge_2


def df_control_general(df):

    # create dataframe for 'Control' group
    df_control = df[df['variation'] == 'Control']
    df_control = df_control.groupby(['variation','date','month','step_check'])['visit_id'].count().reset_index()
    df_control = df_control.pivot(index=['variation','date','month'], columns='step_check', values='visit_id').fillna(0).reset_index()

    return df_control

def df_test_general(df):

    # create dataframe for 'Test' group
    df_test = df[df['variation'] == 'Test']
    df_test = df_test.groupby(['variation','date','month','step_check'])['visit_id'].count().reset_index()
    df_test = df_test.pivot(index=['variation','date','month'], columns='step_check', values='visit_id').fillna(0).reset_index()

    return df_test


def normality_check(df, column_name):

    column = df[column_name]

    # Histogram plot to understand the distribution of data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(column, kde=True, bins=30, color="salmon")
    plt.title(f'Histogram of {column_name}')

    # Q-Q plot 
    plt.subplot(1, 2, 2)
    stats.probplot(column, dist="norm", plot=plt)
    stats.probplot(column, plot=plt)
    plt.title(f'Q-Q Plot of {column_name}')

    plt.tight_layout()
    plt.show()

    # Conducting the Kolmogorov-Smirnov test 
    standardized_column = (column - column.mean()) / column.std()
    ks_test_statistic, ks_p_value = stats.kstest(standardized_column, 'norm')

    ks_test_statistic, ks_p_value

    # print the test result
    if ks_p_value < 0.05:
        print('The test results indicate that the distribution is significantly different from a normal distribution.')
    else:
        print('The test results indicate that the distribution is not significantly different from a normal distribution.')

def data_normalization(df, column_name):

    # transform the data
    log_transformed_column = np.log1p(df[column_name])
    standardized_log_column = (log_transformed_column - log_transformed_column.mean()) / log_transformed_column.std()

    # Plotting histogram for transformed 'column_name'
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(standardized_log_column, kde=True, bins=50, color="skyblue")
    plt.title(f'Histogram of normalized {column_name}')

    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(standardized_log_column, plot=plt)
    plt.title(f'Q-Q Plot of normalized {column_name}')
    
    plt.tight_layout()
    plt.show()

    # Conducting the Kolmogorov-Smirnov test on the log-transformed and standardized column
    ks_test_statistic_after_transformation, ks_p_value_after_transformation = stats.kstest(standardized_log_column, 'norm')

    ks_test_statistic_after_transformation, ks_p_value_after_transformation

    # update the standardized column
    scaler = StandardScaler()
    log_transformed_standardized = scaler.fit_transform(log_transformed_column.values.reshape(-1, 1) ) # standardize log_transformed_column
    
    df[column_name] = scaler.inverse_transform(log_transformed_standardized)


def hypothesis_testing(df1, column_name1, df2, column_name2, alpha):

    # Set Hypothesis

    #H0 total confirmation of test group >= total confirmation of control group
    #H1 total confirmation of test group < total confirmation of control group

    df_confirmation_test = df1[column_name1]
    df_confirmation_control = df2[column_name2]

    t_stat, pvalue = st.ttest_ind(df_confirmation_test,df_confirmation_control, alternative="less")

    print('pvalue is', pvalue)

    if pvalue < alpha:
        print("Fail to reject null hypothesis.")
    else:
        print("Reject null hypothesis.")



""" Natalia's functions """


# (Nat)f_1.1.Perform initial data checking:
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


# (Nat)f_1.8.Histogram for visual inspection of distribution and outliers
def histogram_with_density_plot(df, column):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    # Use the 'column' parameter to reference the DataFrame column
    sns.histplot(df[column], bins=100, kde=True)
    plt.title(f'Histogram with Density Plot for {column}')
    plt.show()


# (Nat)f_1.9.Boxplot to visually check for outliers
def plot_boxplot(df, column):
    # Setting the figure size for better visibility
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot for {column}')

    plt.tight_layout()  # Adjusts plot parameters for better layout
    plt.show()


# (Nat)f_2.0.Shapiro-Wilk test for normality
def shapiro_wilk_test(df, column):
    # Extract the column data
    data = df[column]

    # Perform the Shapiro-Wilk test
    stat, p = stats.shapiro(data)

    # Display the results
    print(f'Shapiro-Wilk Test for {column}: Statistics={stat:.3f}, p={p:.3f}')
    if p > 0.05:
        print("The data appears to be normally distributed (fail to reject H0).")
    else:
        print("The data does not appear to be normally distributed (reject H0).")


# (Nat)f_2.1.0. Load and merge data
def load_and_merge_datasets():
    df_demo = pd.read_csv('../data/cleaned/df_final_demo_cleaned.csv')
    df_experiment = pd.read_csv(
        '../data/cleaned/df_final_experiment_clients_cleaned.csv')
    df_web = pd.read_csv('../data/cleaned/df_final_web_data_cleaned.csv')
    merged = merge_3(df_demo, df_experiment, df_web)
    return merged


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


# (Nat)f_2.10.1.Hypothesis testing for Completion Rate: 'Test' group has completion rate from the A/B test more than 5% then 'Control' group.
def perform_two_sample_t_test(df):
    # Extract the completion rates for Control and Test groups
    control_rate = df[df['Variation'] ==
                      'Control']['average_completion_rate'].iloc[0]
    test_rate = df[df['Variation'] ==
                   'Test']['average_completion_rate'].iloc[0]

    # Hypothetical standard deviations and sample sizes if not given
    std_dev_control = 0.05  # Example standard deviation for Control
    std_dev_test = 0.05     # Example standard deviation for Test
    n_control = 100         # Example sample size for Control
    n_test = 100            # Example sample size for Test

    # Calculate the standard error of the difference between the two means
    se_difference = ((std_dev_control**2 / n_control) +
                     (std_dev_test**2 / n_test))**0.5

    # Calculate the T-statistic for the difference between the groups
    # Subtract 0.05 to test for more than 5% difference
    t_stat = (test_rate - control_rate - 0.05) / se_difference

    # Calculate degrees of freedom for two sample t-test
    df = n_control + n_test - 2

    # Calculate the p-value from the t distribution
    p_value = 1 - stats.t.cdf(t_stat, df)

    # Interpretation based on p-value
    alpha = 0.05  # significance level
    if p_value < alpha:
        interpretation = ("Reject the null hypothesis: There is significant evidence that " +
                          "the 'Test' group's completion rate is more than 5% higher than the 'Control' group.")
    else:
        interpretation = ("Fail to reject the null hypothesis: There is not significant evidence that " +
                          "the 'Test' group's completion rate is more than 5% higher than the 'Control' group.")

    return {'T-statistic': t_stat, 'P-value': p_value, 'Interpretation': interpretation}


# (Nat)f_2.11.Calculate whether the average age of clients engaging with the new process is the same as those engaging with the old process
def calculate_completion_rate_age(df):
    # Split the dataframe into 'Test' and 'Control' groups
    test_group = df[df['variation'] == 'Test']
    control_group = df[df['variation'] == 'Control']

    def get_completion_rate(group):
        # Get the number of unique visit_ids where process_step is 'confirm'
        completed_visits = group[group['process_step']
                                 == 'confirm']['visit_id'].nunique()
        # Get the total number of unique visit_ids in the group
        total_visits = group['visit_id'].nunique()
        # Calculate the completion rate
        completion_rate = completed_visits / total_visits if total_visits > 0 else 0
        # Calculate average age of the group
        average_age = group['age'].mean()
        return completion_rate, total_visits, completed_visits, average_age

    # Get results for both groups
    test_results = get_completion_rate(test_group)
    control_results = get_completion_rate(control_group)

    # Calculate the increase in average age between Test and Control groups
    increase_in_age = (test_results[3] - control_results[3])*100 / \
        control_results[3] if control_results[3] > 0 else 0

    # Create a DataFrame for visualization
    completion_rates_df = pd.DataFrame({
        'variation': ['Control', 'Test'],
        'completion_rate': [control_results[0], test_results[0]],
        'completed_visits': [control_results[2], test_results[2]],
        'total_visits': [control_results[1], test_results[1]],
        'average_age': [control_results[3], test_results[3]]
    })

    return completion_rates_df, increase_in_age


# (Nat)f_2.12.Check if the average client tenure (how long they've been with Vanguard) of those engaging with the new process is the same as those engaging with the old process
def calculate_completion_rate_tenure_months(df):
    # Split the dataframe into 'Test' and 'Control' groups
    test_group = df[df['variation'] == 'Test']
    control_group = df[df['variation'] == 'Control']

    def get_completion_rate(group):
        # Get the number of unique visit_ids where process_step is 'confirm'
        completed_visits = group[group['process_step']
                                 == 'confirm']['visit_id'].nunique()
        # Get the total number of unique visit_ids in the group
        total_visits = group['visit_id'].nunique()
        # Calculate the completion rate
        completion_rate = completed_visits / total_visits if total_visits > 0 else 0
        # Calculate average tenure in months of the group
        average_tenure = group['tenure_month'].mean()
        return completion_rate, total_visits, completed_visits, average_tenure

    # Get results for both groups
    test_results = get_completion_rate(test_group)
    control_results = get_completion_rate(control_group)

    # Calculate the increase in average tenure months between Test and Control groups
    increase_in_tenure = (test_results[3] - control_results[3]) / \
        control_results[3] if control_results[3]*100 > 0 else 0

    # Create a DataFrame for visualization
    completion_rates_df = pd.DataFrame({
        'variation': ['Control', 'Test'],
        'completion_rate': [control_results[0], test_results[0]],
        'average_tenure_months': [control_results[3], test_results[3]]
    })

    return completion_rates_df, increase_in_tenure


# (Nat)f_2.13.Check if there are gender differences that affect engaging with the new or old process
def calculate_completion_rate_gender(df):
    # Split the dataframe into 'Test' and 'Control' groups
    test_group = df[df['variation'] == 'Test']
    control_group = df[df['variation'] == 'Control']

    def get_gender_completion_rate(group, gender):
        # Filter the group by gender
        gender_group = group[group['gender'] == gender]
        # Get the number of unique visit_ids where process_step is 'confirm'
        completed_visits = gender_group[gender_group['process_step']
                                        == 'confirm']['visit_id'].nunique()
        # Get the total number of unique visit_ids in the gender-specific group
        total_visits = gender_group['visit_id'].nunique()
        # Calculate the completion rate
        completion_rate = (completed_visits /
                           total_visits) if total_visits > 0 else 0
        return completion_rate

    # Create a DataFrame for visualization with structured columns
    data = [
        {'variation': 'Test', 'gender': 'Male',
            'completion_rate': get_gender_completion_rate(test_group, 'M')},
        {'variation': 'Test', 'gender': 'Female',
            'completion_rate': get_gender_completion_rate(test_group, 'F')},
        {'variation': 'Control', 'gender': 'Male',
            'completion_rate': get_gender_completion_rate(control_group, 'M')},
        {'variation': 'Control', 'gender': 'Female',
            'completion_rate': get_gender_completion_rate(control_group, 'F')}
    ]
    completion_rates_df = pd.DataFrame(data)

    # Calculate the increase in completion rates
    increase_male = data[0]['completion_rate'] - data[2]['completion_rate']
    increase_female = data[1]['completion_rate'] - data[3]['completion_rate']

    # Print increases separately for analysis
    print(f"Increase in completion rate for males (Test vs Control): {
          increase_male:.2%}")
    print(f"Increase in completion rate for females (Test vs Control): {
          increase_female:.2%}")

    return completion_rates_df

    # Display increases separately
    print(f"Increase in completion rate for males (Test vs Control): {
          increase_male:.2%}")
    print(f"Increase in completion rate for females (Test vs Control): {
          increase_female:.2%}")

    return completion_rates_df


"""Natalia's Functions related to Time spent per visit"""


# (Nat)f_3.1.Calculate time spent per visit and determine if the visit was completed
def calculate_time_and_completion(df):
    # Creating the 'completed_yes_no' column
    df['completed_yes_no'] = df['process_step'].apply(
        lambda x: 1 if x == 'confirm' else 0)

    # Grouping by 'visit_id' to calculate the time spent for each visit
    df['time_per_visit_in_sec'] = df.groupby('visit_id')['date_time'].transform(
        lambda x: (x.max() - x.min()).total_seconds())

    # Converting the time difference to seconds and changing the data type to int64
    df['time_per_visit_in_sec'] = df['time_per_visit_in_sec'].astype('int64')

    # Dropping duplicate rows based on 'visit_id' while keeping the first occurrence
    df = df.drop_duplicates(subset='visit_id')

    return df


# (Nat)f_3.2.Check if the time spent for each visit is normally distributed with Histogram and Boxplot to visually check for outliers
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


# (Nat)f_3.3.Shapiro-Wilk test to check if data time_per_visit is normally distributed
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


# (Nat)f_3.6.calculate the average time spent per visit_id for a given variation (after cleaning outliers)
def average_time_spent_per_variation_and_segment(df, column):
    # Gets all unique variations from the specified column
    variations = df[column].unique()
    gender_results = []
    age_results = []
    tenure_results = []

    for variation in variations:
        filtered_df = df[df[column] == variation]

        # Calculate average time by gender within each variation
        gender_avg = filtered_df.groupby(
            'gender')['time_per_visit_in_sec'].mean().reset_index()
        gender_avg['Variation'] = variation
        gender_avg.rename(
            columns={'time_per_visit_in_sec': 'Average Time (seconds)'}, inplace=True)
        gender_results.append(gender_avg)

        # Calculate average time by age within each variation
        age_avg = filtered_df.groupby(
            'age')['time_per_visit_in_sec'].mean().reset_index()
        age_avg['Variation'] = variation
        age_avg.rename(
            columns={'time_per_visit_in_sec': 'Average Time (seconds)'}, inplace=True)
        age_results.append(age_avg)

        # Calculate average time by tenure_month within each variation
        tenure_avg = filtered_df.groupby('tenure_month')[
            'time_per_visit_in_sec'].mean().reset_index()
        tenure_avg['Variation'] = variation
        tenure_avg.rename(
            columns={'time_per_visit_in_sec': 'Average Time (seconds)'}, inplace=True)
        tenure_results.append(tenure_avg)

    # Convert results lists to DataFrames
    df_gender_results = pd.concat(gender_results).reset_index(drop=True)
    df_age_results = pd.concat(age_results).reset_index(drop=True)
    df_tenure_results = pd.concat(tenure_results).reset_index(drop=True)

    # Export to CSV files
    df_gender_results.to_csv(
        '../data/cleaned/average_time_by_gender.csv', index=False)
    df_age_results.to_csv(
        '../data/cleaned/average_time_by_age.csv', index=False)
    df_tenure_results.to_csv(
        '../data/cleaned/average_time_by_tenure.csv', index=False)

    return df_gender_results, df_age_results, df_tenure_results


# (Nat)f_3.7. count_visit_ids_per_process_step. % of dropped users on each step.
def count_visit_ids_per_process_step(df):
    # Define the process steps of interest with 'confirm' being the last one
    process_steps = ['start', 'step_1', 'step_2', 'step_3', 'confirm']

    # Filter the dataframe to include only the rows with the specified process steps
    filtered_df = df[df['process_step'].isin(process_steps)]

    # Ensure the process steps are ordered correctly in the DataFrame
    filtered_df['process_step'] = pd.Categorical(
        filtered_df['process_step'], categories=process_steps, ordered=True)

    # Group by 'variation' and 'process_step', and count distinct 'visit_id' in each group
    visit_counts = (filtered_df.groupby(['variation', 'process_step'])['visit_id']
                    .nunique()
                    .reset_index(name='visit_count'))

    # Calculate the total percentage drop within each 'variation'
    result = []

    for variation in visit_counts['variation'].unique():
        # Filter for the current variation
        variation_df = visit_counts[visit_counts['variation']
                                    == variation].copy()

        # Get the count at the 'start' step
        start_count = variation_df[variation_df['process_step']
                                   == 'start']['visit_count'].values
        if len(start_count) > 0:
            start_count = start_count[0]
            variation_df['percentage'] = (
                variation_df['visit_count'] / start_count) * 100

            # Calculate the drop percentage relative to the previous step
            variation_df['drop_percentage'] = variation_df['visit_count'].pct_change(
            ) * 100

            # Set the drop percentage for the 'start' step to 0%
            variation_df.loc[variation_df['process_step']
                             == 'start', 'drop_percentage'] = 0

            # Round the percentage and drop percentage columns to 2 decimal places
            variation_df['percentage'] = variation_df['percentage'].round(2)
            variation_df['drop_percentage'] = variation_df['drop_percentage'].round(
                2)

        # Calculate the total drop percentage from 'start' to 'confirm'
        total_drop = (variation_df[variation_df['process_step'] == 'confirm']['visit_count'].values
                      / start_count) * 100 - 100
        total_drop = total_drop.round(2)

        # Add a row for the total drop percentage
        total_drop_row = pd.DataFrame({
            'variation': [variation],
            'process_step': ['total_drop'],
            'visit_count': [None],
            'percentage': [None],
            'drop_percentage': [total_drop]
        })

        # Append the result for the current variation
        result.append(
            pd.concat([variation_df, total_drop_row], ignore_index=True))

    # Concatenate results for all variations into a single DataFrame
    final_df = pd.concat(result).reset_index(drop=True)

    return final_df


# (Nat)f_3.8. How the call center is solicited in each group? by which segments of users (age, gender) ?
def total_calls_per_group_gender(df):
    # Group by 'variation' and calculate the total calls in 'calls_6_month'
    total_calls_variation = df.groupby(
        'variation')['calls_6_month'].sum().reset_index()

    # Calculate the percentage increase between 'Test' and 'Control' for overall calls
    control_calls = total_calls_variation[total_calls_variation['variation']
                                          == 'Control']['calls_6_month'].values[0]
    test_calls = total_calls_variation[total_calls_variation['variation']
                                       == 'Test']['calls_6_month'].values[0]
    overall_percentage_increase = (
        (test_calls - control_calls) / control_calls) * 100

    # Group by 'variation' and 'gender' and calculate the total calls in 'calls_6_month'
    total_calls_gender = df.groupby(['variation', 'gender'])[
        'calls_6_month'].sum().reset_index()

    # Calculate the percentage increase between 'Test' and 'Control' for each gender
    gender_percentage_increase = {}
    genders = df['gender'].unique()
    for gender in genders:
        control_calls_gender = total_calls_gender[(total_calls_gender['variation'] == 'Control') & (
            total_calls_gender['gender'] == gender)]['calls_6_month'].values
        test_calls_gender = total_calls_gender[(total_calls_gender['variation'] == 'Test') & (
            total_calls_gender['gender'] == gender)]['calls_6_month'].values
        # Ensure there is data for both groups
        if len(control_calls_gender) > 0 and len(test_calls_gender) > 0:
            control_calls_gender = control_calls_gender[0]
            test_calls_gender = test_calls_gender[0]
            gender_percentage_increase[gender] = (
                (test_calls_gender - control_calls_gender) / control_calls_gender) * 100

    # Create DataFrames for display and export
    df_total_calls_variation = pd.DataFrame(total_calls_variation)
    df_total_calls_gender = pd.DataFrame(total_calls_gender)
    df_gender_percentage_increase = pd.DataFrame(list(
        gender_percentage_increase.items()), columns=['Gender', 'Percentage Increase'])
    df_overall_percentage_increase = pd.DataFrame({
        'Comparison': ['Overall'],
        'Percentage Increase': [overall_percentage_increase]
    })

    # Export DataFrames to CSV
    df_total_calls_variation.to_csv(
        '../data/cleaned/total_calls_variation.csv', index=False)
    df_total_calls_gender.to_csv(
        '../data/cleaned/total_calls_gender.csv', index=False)
    df_gender_percentage_increase.to_csv(
        '../data/cleaned/gender_calls_percentage_increase.csv', index=False)
    df_overall_percentage_increase.to_csv(
        '../data/cleaned/overall_calls_percentage_increase.csv', index=False)

    return df_total_calls_variation, df_total_calls_gender, df_gender_percentage_increase, df_overall_percentage_increase


# (Nat)f_3.9. How often the user in each group logons_6_month ? by which segments of users (gender) ?
def total_logons_per_group_gender(df):
    # Group by 'variation' and calculate the total logons in 'logons_6_month'
    total_logons_variation = df.groupby(
        'variation')['logons_6_month'].sum().reset_index()

    # Calculate the percentage increase between 'Test' and 'Control' for overall logons
    control_logons = total_logons_variation[total_logons_variation['variation']
                                            == 'Control']['logons_6_month'].values[0]
    test_logons = total_logons_variation[total_logons_variation['variation']
                                         == 'Test']['logons_6_month'].values[0]

    overall_percentage_increase = (
        (test_logons - control_logons) / control_logons) * 100

    # Group by 'variation' and 'gender' and calculate the total logons in 'logons_6_month'
    total_logons_gender = df.groupby(['variation', 'gender'])[
        'logons_6_month'].sum().reset_index()

    # Calculate the percentage increase between 'Test' and 'Control' for each gender
    gender_percentage_increase = {}
    genders = df['gender'].unique()
    for gender in genders:
        control_logons_gender = total_logons_gender[(total_logons_gender['variation'] == 'Control') & (
            total_logons_gender['gender'] == gender)]['logons_6_month'].values
        test_logons_gender = total_logons_gender[(total_logons_gender['variation'] == 'Test') & (
            total_logons_gender['gender'] == gender)]['logons_6_month'].values
        # Ensure there is data for both groups
        if len(control_logons_gender) > 0 and len(test_logons_gender) > 0:
            control_logons_gender = control_logons_gender[0]
            test_logons_gender = test_logons_gender[0]
            gender_percentage_increase[gender] = (
                (test_logons_gender - control_logons_gender) / control_logons_gender) * 100

        # Export dataframes to CSV
        total_logons_gender.to_csv(
            '../data/cleaned/logons_by_gender.csv', index=False)

    return total_logons_variation, total_logons_gender, overall_percentage_increase, gender_percentage_increase


""" Tobias' Functions """

""" Functions for Bivariate EDA """


def create_correlation_matrix(df, cols_numerical):

    correlation_matrix = df[cols_numerical].corr()

    # Setting up the matplotlib figure with an appropriate size
    plt.figure(figsize=(6, 5))

    # Drawing the heatmap for the numerical columns
    sns.heatmap(round(correlation_matrix, 2), annot=True,
                cmap="coolwarm", vmin=-1, vmax=1)

    plt.title("Correlation Heatmap for Selected Numerical Variables")
    plt.show()


def plot_balance_vs_age(df):
    df_age_bal = pd.DataFrame({'age': pd.pivot_table(df, index="age", values="balance", aggfunc='mean').index,
                               'balance_mean': pd.pivot_table(df, index="age", values="balance", aggfunc='mean').balance})

    fig, ax = plt.subplots()
    sns.scatterplot(df, x="age", y="balance", ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(df_age_bal, x="age", y="balance_mean", ax=ax2, color='orange')
    fig.legend(labels=['balance', 'average balance'], bbox_to_anchor=(
        0.15, 0.85), loc='upper left', borderaxespad=0)
    plt.show()


""" Functions for Experiment Evaluation """


def experiment_evalutaion(df):
    df["is_female"] = df["gender"].apply(lambda x: True if x == "F" else False)
    df_pivot = df.pivot_table(index='variation',
                              values=['age', 'tenure_year', 'number_of_accounts',
                                      'balance', 'calls_6_month', 'logons_6_month', 'is_female'],
                              aggfunc="mean")
    print('Bias Test vs Control: ')
    variables = []
    biases = []
    for col in df_pivot.columns:
        bias = df_pivot.loc["Test"][col]/df_pivot.loc["Control"][col]-1
        variables.append(col)
        biases.append(bias)
    return pd.DataFrame({"variable": variables, "bias": biases})


def calculate_avg_daily_visits_per_time_period(df):
    visits_total = (df["Control"] + df["Test"])

    visits_1 = visits_total.loc[(visits_total.index <= 13)].mean()
    visits_2 = visits_total.loc[(visits_total.index <= 41) & (
        visits_total.index > 13)].mean()
    visits_3 = visits_total.loc[(visits_total.index > 41)].mean()

    print(f"Average Daily Visits")
    print(f"--------------------")
    print(f"In the last two weeks of March: {int(round(visits_1, 0))}")
    print(f"In April: {int(round(visits_2, 0))}")
    print(f"In May and June: {int(round(visits_3, 0))}")


""" Functions for Error Rates """


def calculate_avg_errors_per_visit(df):
    df_visits = df[['unique_session_id', 'process_step', 'date_time',
                    'variation']].sort_values(by=["unique_session_id", "date_time"])
    process_step_dict = {'start': 0, 'step_1': 1,
                         'step_2': 2, 'step_3': 3, 'confirm': 4}
    df_visits['process_step_number'] = df_visits['process_step'].map(
        process_step_dict)
    df_visits['previous_step_number'] = df_visits.groupby(
        'unique_session_id')['process_step_number'].shift()
    df_visits['step_diff'] = df_visits['process_step_number'] - \
        df_visits['previous_step_number']
    df_visits['step_back'] = df_visits['step_diff'].apply(
        lambda x: True if x < 0 else False)
    total_errors_control = df_visits.loc[(
        df_visits["variation"] == "Control")]['step_back'].sum()
    total_errors_test = df_visits.loc[(
        df_visits["variation"] == "Test")]['step_back'].sum()
    total_visits_control = df_visits.loc[(
        df_visits["variation"] == "Control")]["unique_session_id"].nunique()
    total_visits_test = df_visits.loc[(
        df_visits["variation"] == "Test")]["unique_session_id"].nunique()
    error_rate_control = total_errors_control / total_visits_control
    error_rate_test = total_errors_test / total_visits_test
    return error_rate_control, error_rate_test


def calculate_avg_error_rates_per_time_period(df):

    error_rate_control_1, error_rate_test_1 = calculate_avg_errors_per_visit(
        df)
    error_rate_control_2, error_rate_test_2 = calculate_avg_errors_per_visit(
        df.loc[df["day_of_trial"] < 55])
    error_rate_control_3, error_rate_test_3 = calculate_avg_errors_per_visit(
        df.loc[df["day_of_trial"] >= 55])

    print(f"Daily Average Error Rates")
    print(f"-------------------------")
    print(f"")
    print(f"In total")
    print(f"Control: {round(error_rate_control_1*100, 1)}%")
    print(f"Test: {round(error_rate_test_1*100, 1)}%")
    print(f"")
    print(f"Trial Day < 55")
    print(f"Control: {round(error_rate_control_2*100, 1)}%")
    print(f"Test: {round(error_rate_test_2*100, 1)}%")
    print(f"")
    print(f"Trial Day >= 55")
    print(f"Control: {round(error_rate_control_3*100, 1)}%")
    print(f"Test: {round(error_rate_test_3*100, 1)}%")


def calculate_grouped_error_rates(df, grouping_column):
    error_rates = pd.DataFrame({"error_rate_control": df.groupby([grouping_column]).apply(lambda x: calculate_avg_errors_per_visit(x)[0], include_groups=False),
                                "error_rate_test": df.groupby([grouping_column]).apply(lambda x: calculate_avg_errors_per_visit(x)[1], include_groups=False)})
    return error_rates


def plot_avg_errors_per_day(df):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    sns.scatterplot(df, ax=axs[0, 0])
    axs[0, 0].set_title('Average Error Rate per Day')
    axs[0, 0].set_xlabel('Day of Trial')
    axs[0, 0].set_ylabel('Error Rate')

    sns.histplot(df, bins=20, ax=axs[0, 1])
    axs[0, 1].set_title('Average Daily Error Rate Distribution')
    axs[0, 1].set_xlabel('Errors Rate')
    axs[0, 1].set_ylabel('Count')

    # Distribution < Day 55
    sns.histplot(df.loc[(df.index < 55)], bins=15, ax=axs[1, 0])
    axs[1, 0].set_title('Trial Day < 55')
    axs[1, 0].set_xlabel('Errors Rate')
    axs[1, 0].set_ylabel('Count')

    # Distribution >= Day 55
    sns.histplot(df.loc[(df.index >= 55)], bins=15, ax=axs[1, 1])
    axs[1, 1].set_title('Trial Day >= 55')
    axs[1, 1].set_xlabel('Errors Rate')
    axs[1, 1].set_ylabel('Count')

    fig.tight_layout()
    plt.show()


def errors_daily_to_csv(df):
    # bring data into correct format for PowerBI
    errors_daily_control = pd.DataFrame(
        {"trial_day": df.index, "error_rate": df["error_rate_control"], "variation": "Control"})
    errors_daily_test = pd.DataFrame(
        {"trial_day": df.index, "error_rate": df["error_rate_test"], "variation": "Test"})
    errors_daily_csv = pd.concat([errors_daily_control, errors_daily_test],
                                 axis=0, join='inner', ignore_index=True)  # default 'outer'

    # save average errors per day to csv-file
    errors_daily_csv.to_csv(
        "../data/visualization/errors_daily.csv", index=True, decimal=',', encoding='utf-8')


def error_rates_hypothesis_test_vs_control(df):

    _, p_value_0 = stats.ttest_ind(df.loc[(df.index)]['error_rate_control'],
                                   df.loc[(df.index)]['error_rate_test'], equal_var=False, alternative='less')

    _, p_value_1 = stats.ttest_ind(df.loc[(df.index < 55)]['error_rate_control'],
                                   df.loc[(df.index < 55)]['error_rate_test'], equal_var=False, alternative='less')

    _, p_value_2 = stats.ttest_ind(df.loc[(df.index >= 55)]['error_rate_control'],
                                   df.loc[(df.index >= 55)]['error_rate_test'], equal_var=False, alternative='less')

    print(f"p-values for H0")
    print(f"---------------")
    print(f"All Trial Days: {p_value_0:.{1}e}")
    print(f"Trial Day < 55: {p_value_1:.{1}e}")
    print(f"Trial Day >= 55: {p_value_2:.{1}e}")


def error_rates_hypothesis_early_vs_late_test(df):
    # Compare test group before day 55 and after
    _, p_value = stats.ttest_ind(df.loc[(df.index < 55)]['error_rate_test'],
                                 df.loc[(df.index >= 55)]['error_rate_test'], equal_var=False)
    print(f"p-value for H0: {p_value:.{1}e}")


def error_rates_hypothesis_early_vs_late_control(df):
    # Compare test group before day 55 and after
    _, p_value = stats.ttest_ind(df.loc[(df.index < 55)]['error_rate_control'],
                                 df.loc[(df.index >= 55)]['error_rate_control'], equal_var=False)
    print(f"p-value for H0: {p_value:.{1}e}")
