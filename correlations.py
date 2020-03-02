import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_series(df):
    result = seasonal_decompose(df, model='additive', freq=365)
    result.plot()
    plt.show()

def find_correlation(df, col1, col2):
    # Compute percent change using pct_change()
    df[col1+'_ret'] = df[col1].pct_change()
    df[col2+'_ret'] = df[col2].pct_change()

    # Compute correlation using corr()
    correlation_df = df[col1].corr(df[col2])
    correlation_ret_df = df[col1+'_ret'].corr(df[col2+'_ret'])
    print("Correlation of "+col1+" and "+col2+": ", correlation_df)
    print("Correlation (Percent Changes) of "+col1+" and "+col2+": ", correlation_ret_df)

    # Make scatter plot
    # plt.scatter(df[col1+'_ret'], df[col2+'_ret'])
    # plt.show()

def find_autocorrelation(df, col):
    # Convert the daily data to weekly data
    # df = df.resample(rule='W', convention='end')
    df[col] = df[col].resample('W').sum()
    df[col+'_ret'] = df[col].pct_change()

    # Compute and print the autocorrelation of returns
    autocorrelation = df[col+'_ret'].autocorr()
    print("The autocorrelation of weekly returns is %4.2f" % (autocorrelation))

# Twitter data with transaction counts irrespective of stores.
def twitter_analysis():
    df_sales_decompose = pd.read_excel('data/sales.xlsx', header=0, index_col=0, usecols=['Date', 'TransactionCount'])

    xlsx = pd.ExcelFile('data/sales.xlsx')
    df_sales = pd.read_excel(xlsx, 'sheet1')
    df_sales = df_sales[['Date', 'TransactionCount']]
    df_sales['Date'] = pd.to_datetime(df_sales.Date, yearfirst=True)
    # df_sales_group = df_sales['TransactionCount'].groupby(df_sales['Date']).sum()
    df_sales_group = df_sales.groupby(['Date'])[["TransactionCount"]].sum()

    df_twitter = pd.read_csv('data/Twitter Profiles-01-01-2018-02-09-2020.csv')
    df_twitter = df_twitter[['Date', 'Impressions', 'Published Posts']]
    df_twitter['Date'] = pd.to_datetime(df_twitter.Date, format="%m-%d-%Y")
    # df_twitter.plot(grid=True)
    # plt.show()

    # Correlation of Two Time Series
    df = df_twitter.join(df_sales_group, on='Date', how='inner')
    df.Impressions = df['Impressions'].str.replace(',', '')
    df.Impressions = pd.to_numeric(df['Impressions'])
    df.TransactionCount = pd.to_numeric(df['TransactionCount'])
    df.index=df['Date']
    # print(df.info())

    reggression = scipy.stats.linregress(df['Published Posts'], df['Impressions'])
    print("Standard error is ", reggression.stderr)

    # find_correlation(df, 'Impressions', 'TransactionCount');
    # find_correlation(df, 'Published Posts', 'Impressions');
    # find_autocorrelation(df, 'Impressions')




if __name__ == '__main__':
    twitter_analysis()