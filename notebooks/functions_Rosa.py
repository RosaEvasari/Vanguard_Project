import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import scipy.stats as st
import numpy as np
from statsmodels.stats.proportion import proportions_ztest # pip install statsmodels
from sklearn.preprocessing import StandardScaler

# Read csv file

def read_txt(file_path):
    
    file = pd.read_csv(file_path, delimiter=',') 
    df = pd.DataFrame(file)
    df = df.copy()

    return df


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


def univariate_categorical(df):

    # check the frequency 
    frequency_table = df.value_counts()

    # visualization
    
    sns.barplot(x=frequency_table.index, y=frequency_table.values, palette='viridis')
    plt.title('Visualization of The Distribution')
    plt.xticks(ha='right')
    plt.show()


def univariate_numerical(df):

    # Measure of centrality
    mean = round(df.mean(),2)
    median = round(df.median(),2)
    mode = round(df.mode()[0],2)

    # Measure of dispersion
    variance = round(df.var(),2)
    std_dev = round(df.std(),2)
    min_value = df.min()
    max_value = df.max()
    range_value = max_value - min_value
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    quantiles = df.quantile([0.25, 0.5, 0.75])

    # Shape of distribution
    skewness = round(df.skew(),2)
    kurtosis = round(df.kurtosis(),2)

    summary_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Mode', 'Variance', 'Standard Deviation', 'Min Value', 'Max Value', 'Range', 'Interquartile Range', 'Skewness', 'Kurtosis'],
        'Value': [mean, median, mode, variance, std_dev, min_value, max_value, range_value, IQR, skewness, kurtosis]
    })


    # Visualization

    # Histogram plot
    plt.subplot(2, 1, 1)
    sns.histplot(df, kde=True, bins=20, color="skyblue")
    plt.title('Histogram plot')

    # Box plot
    plt.subplot(2, 1, 2)
    sns.boxplot(data = df, color="skyblue")
    plt.title('Box Plot')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust the space between plots

    plt.show()
    
    return summary_df


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