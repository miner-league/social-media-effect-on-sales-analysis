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


def get_row_data(data, level, column_name, ID):
    store_data = data[data[column_name] == ID]
    data_without_score = store_data[[
        'Date',
        column_name,
        'SmoothedTransactionCount'
    ]]
    data_with_just_score = store_data[[
        'Date',
        'Weighted Social Medial Activity Score'

    ]]
    data_without_score = data_without_score.groupby(['Date', column_name], as_index=False).sum()

    store_data = pd.merge(
        data_without_score,
        data_with_just_score,
        how='left',
        on='Date'
    )

    weighted_scores = store_data['Weighted Social Medial Activity Score']
    smoothed_transactions = store_data['SmoothedTransactionCount']

    row = {'Level': level, 'ID': ID}

    number_of_days = 30

    for offset in range(number_of_days, -1, -1):
        smoothed_transactions = smoothed_transactions.shift(periods=-1, fill_value=np.nan)
        smoothed_transactions = smoothed_transactions.dropna()
        correlation = weighted_scores.corr(smoothed_transactions, method='pearson')
        row[str(number_of_days - offset)] = correlation

    return row


# def get_row_for_city(data, column, city):
#     store_data = data[data[column] == city]
#     store_data = store_data[['Date', 'StoreId', 'Weighted Social Medial Activity Score', 'SmoothedTransactionCount']]
#
#     weighted_scores = store_data['Weighted Social Medial Activity Score']
#     smoothed_transactions = store_data['SmoothedTransactionCount']
#
#     row = {'Level': 'City', 'ID': city}
#
#     number_of_days = 30
#
#     for offset in range(number_of_days, -1, -1):
#         smoothed_transactions = smoothed_transactions.shift(periods=-1, fill_value=np.nan)
#         smoothed_transactions = smoothed_transactions.dropna()
#         correlation = weighted_scores.corr(smoothed_transactions, method='pearson')
#         row[str(number_of_days - offset)] = correlation
#
#     return row


def calculate_social_media_score_sales_correlations():
    social_media_scores = pd.read_csv('data/social_media_activity_scores.csv')
    social_media_scores['Date'] = pd.to_datetime(social_media_scores['Date'])
    sales_with_smoothing = pd.read_csv('data/sales_with_smoothing.csv')
    sales_with_smoothing['Date'] = pd.to_datetime(sales_with_smoothing['Date'])

    stores = pd.read_excel('data/stores.xlsx')

    combined = pd.merge(
        sales_with_smoothing,
        social_media_scores,
        how='left',
        on='Date'
    )

    combined = pd.merge(
        combined,
        stores,
        how='left',
        on='StoreId'
    )

    combined = combined.dropna()

    rows = []

    regions = stores['Region'].unique()
    markets = stores['Market'].unique()
    states = stores['State'].unique()
    cities = stores['City'].unique()
    store_ids = stores['StoreId'].unique()

    for region in regions:
        rows.append(get_row_data(combined, 'Region', 'Region', region))

    for market in markets:
        rows.append(get_row_data(combined, 'Market', 'Market', market))

    for state in states:
        rows.append(get_row_data(combined, 'State', 'State', state))

    for city in cities:
        rows.append(get_row_data(combined, 'City', 'City', city))

    for store_id in store_ids:
        rows.append(get_row_data(combined, 'Store', 'StoreId', store_id))

    df = pd.DataFrame(rows)

    df.to_csv('correlations_between_media_and_transactions.csv', index=False)


if __name__ == '__main__':
    twitter_analysis()