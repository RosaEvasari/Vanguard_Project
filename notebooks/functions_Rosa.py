import pandas as pd
import plotly.express as px


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