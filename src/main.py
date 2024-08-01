print("- import packages")
import sys
import os
import pandas as pd
import numpy as np

print("- import functions")
# include project folder to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.functions import *

if __name__ == "__main__":

    ##### LOAD AND CLEAN RAW DATA #####
    print("- loading, cleaning and saving datasets")

    ## DEMOGRAPHICS TABLE ##
    print("--- demographics table")

    # Load data
    df_demo = pd.read_csv('../data/raw/df_final_demo.txt')

    # Rename columns and explore table
    df_demo = rename_demo_columns(df_demo)

    # Select categorical and numerical columns
    cols_demo_numerical = ['tenure_year', 'tenure_month', 'age', 'number_of_accounts', 'balance', 'calls_6_month', 'logons_6_month']
    cols_demo_categorical = ['gender', 'client_id']

    # Fill missing values in numerical columns with the mean and in categorical columns with the mode
    df_demo = fill_missing(df_demo, cols_demo_numerical, cols_demo_categorical)

    # Change discrete numerical variables from type float to integer + client_id from integer to object
    df_demo[['number_of_accounts', 'calls_6_month', 'logons_6_month']] = df_demo[['number_of_accounts', 'calls_6_month', 'logons_6_month']].astype(("int64"))
    df_demo['client_id'] = df_demo['client_id'].astype(str)

    # Save cleaned dataframe to csv-file
    df_demo.to_csv("../data/cleaned/df_final_demo_cleaned.csv", index=False)


    ## WEB DATA TABLE ##
    print("--- web data table")

    # Load data
    df_web_data_pt_1 = pd.read_csv('../data/raw/df_final_web_data_pt_1.txt')
    df_web_data_pt_2 = pd.read_csv('../data/raw/df_final_web_data_pt_2.txt')

    # Concatenate two parts
    df_web_data = pd.concat([df_web_data_pt_1, df_web_data_pt_2], axis=0)

    # Select categorical and numerical columns
    cols_web_numerical = []
    cols_web_categorical = ['client_id', 'visitor_id', 'visit_id', 'process_step', 'data_time']

    # Change client_id column from type integer to object
    df_web_data['client_id'] = df_web_data['client_id'].astype(str)

    # Change date_time column from type object to datetime and split into date and time
    df_web_data['date_time'] = pd.to_datetime(df_web_data['date_time'])

    # Drop duplicates
    df_web_data = df_web_data.drop_duplicates()

    # Save cleaned dataframe to csv-file
    df_web_data.to_csv("../data/cleaned/df_final_web_data_cleaned.csv", index=False)


    ## EXPERIMENT CLIENTS TABLE ##
    print("--- experiment clients table")

    # Load data
    df_clients = pd.read_csv('../data/raw/df_final_experiment_clients.txt')

    # Rename column and explore table
    df_clients = df_clients.rename(columns={'Variation': 'variation'})

    # Select categorical and numerical columns
    cols_clients_numerical = []
    cols_clients_categorical = ['client_id', 'variation']

    # Change client_id column from type integer to object
    df_clients['client_id'] = df_clients['client_id'].astype(str)

    # Drop all missing values
    df_clients = df_clients.dropna(subset=["variation"])

    # Save cleaned dataframe to csv-file
    df_clients.to_csv("../data/cleaned/df_final_experiment_clients_cleaned.csv", index=False)



    ##### MERGE CLEAN DATAFRAMES #####
    print("- merge cleaned dataframes")

    # merge tables and only consider clients present in all three tables
    df_ = pd.merge(df_web_data, df_clients, on='client_id', how='inner')
    df = pd.merge(df_, df_demo, on='client_id', how='inner')

    # Add extra column indicating the day counting from the start of the trial
    df["day_of_trial"] = df["date_time"].dt.dayofyear - df["date_time"].sort_values().dt.dayofyear.iloc[0]

    # Add extra column to identify really unique sessions
    df['unique_session_id'] = df['client_id'] + '_' + df['visit_id']



    ##### CREATE AND SAVE CSV-FILES FOR THE DASHBOARD #####
    print("- create and save csv-files")

    # Visits and Error Rates per Day
    errors_and_visits_daily_to_csv(df)
