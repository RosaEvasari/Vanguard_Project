import pandas as pd
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go


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
    "clnt_tenure_yr" : "tenure_year",
    "clnt_tenure_mnth": "tenure_month",
    "clnt_age": "age",
    "gendr" : "gender",
    "num_accts": "number_of_accounts",
    "bal" : "balance",
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


def plot_distributions_numerical(df, cols_numerical):
    fig = make_subplots(rows=len(cols_numerical), cols=1, subplot_titles=(cols_numerical), vertical_spacing=0.04)

    k = 0
    for variable in cols_numerical:
        k += 1
        fig.add_trace(go.Histogram(x=df[variable], name=variable), row=k, col=1)

    height= 250*len(cols_numerical)
    fig.update_layout(height=height, width=800)
    fig.show(config={'staticPlot': True})