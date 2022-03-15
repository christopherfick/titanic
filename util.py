import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

def info(obj, search=''):
    obj_contents = [i for i in dir(obj) if '__' not in i]
    if obj_contents:
        search_results = np.array([i for i in obj_contents if search.lower() in i.lower()])
        if search_results.size <= 0:
            print('No mathcing results')
            return np.array([i for i in obj_contents])
        else:
            return search_results
    else:
        return help(obj)
    

def subset_data(df, *names_like):
    return df[[col for col in df if [name for name in names_like if name in col]]]


def numeric_features(df):
    """Leverages pd.DataFrame.describe() returns numeric data only, convert column names to list"""
    return df.describe().columns.to_list()


def outliers_mask(df):
    df = df[numeric_features(df)]
    Q1, Q3 = df.quantile(0.25) ,df.quantile(0.75)
    iqr = Q3 - Q1
    return ((df < (Q1 - 1.5 * iqr)) | (df > (Q3 + 1.5 + iqr)))


def outliers_replace(df, value):
    df = df.copy(deep=True)
    df[outliers_mask(df)] = value
    return df
    
def outliers_count(df):
    return outliers_mask(df).sum()  # True=1, False=0


def outliers_percentage(df):
    total_rows = df.shape[0]
    return round(outliers_count(df) / total_rows * 100, 2)


def outliers_drop(df):
    df = df.copy(deep=True)
    df[outliers_mask(df)] = np.nan
    return df.dropna()


def outliers_describe(df):
    total_rows = df.shape[0]
    outlier_rows = outliers_drop(df).shape[0]
    print(f'''
    Total Rows: {total_rows}
    Outlier Rows: {outlier_rows}
    Overall data reduction: {round(outlier_rows / total_rows * 100, 2)} %
    
    Count Outliers:\n\n{outliers_count(df)}
    \n________________________________________________________________________
    
    Percentage Outliers:\n\n{outliers_percentage(df)}''')
    
    
def shift_column(df, column_name, loc=None):
    df = df.copy(deep=True)
    if not loc:
        # Move column to end of DataFrame
        loc = len(df.columns) -1
    column_to_move = df.pop(column_name)
    df.insert(loc, column_name)
    return df


def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(12,12))
    heatmap = sns.heatmap(df.corr(), ax=ax)
    heatmap.set_title('Correlation Heatmap', 
                      fontdict={'fontsize':12},
                      pad=12);

    
def plot_model_performance(actual, predictions, title=''):
    plt.plot(actual, predictions, '.')
    plt.plot(actual, actual, 'r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    if title:
        plt.title(title)
    plt.show()
    

def regression_model_metrics(actual, predictions, name=''):
    mse = round(mean_squared_error(actual, predictions), 2)
    rmse = round(np.sqrt(mse), 2)
    r2 = round(r2_score(actual, predictions), 2)
    print(f'\
    {name} MSE: {mse} \n\
    {name} RMSE: {rmse} \n\
    {name} R2_score: {round(r2 * 100, 2)} %')
    

def list_correlations(df, ascending=False):
    df = df.copy(deep=True)
    df_corr = df.corr().abs()
    np.fill_diagonal(df_corr.values, np.nan)

    return df_corr.unstack().dropna().sort_values(ascending=ascending)