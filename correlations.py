import csv
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose_series(df):
    result = seasonal_decompose(df, model='additive', freq=365)
    result.plot()
    plt.show()

def find_correlation(lag, type, df, col1, col2):
    df.index = df['Date']

    # Compute percent change using pct_change()
    df[col1+'_ret'] = df[col1].pct_change()
    df[col2+'_ret'] = df[col2].pct_change()

    # Compute correlation using corr()
    correlation = df[col1].corr(df[col2])
    correlation_percent = df[col1+'_ret'].corr(df[col2+'_ret'])
    print("Correlation of "+col1+" and "+col2+": ", correlation)
    print("Correlation (Percent Changes) of "+col1+" and "+col2+": ", correlation_percent)
    newrow = {
        'Type': type,
        'Lag': lag,
        'Correlation': correlation,
        'CorrelationPercent': correlation_percent
    }
    return newrow

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
def correlation_analysis():
    # df_sales_decompose = pd.read_excel('data/sales.xlsx', header=0, index_col=0, usecols=['Date', 'TransactionCount'])
    #
    # xlsx = pd.ExcelFile('data/sales.xlsx')
    # df_sales = pd.read_excel(xlsx, 'sheet1')
    # df_sales = df_sales[['Date', 'TransactionCount']]
    # df_sales['Date'] = pd.to_datetime(df_sales.Date, yearfirst=True)
    # df_sales_group = df_sales['TransactionCount'].groupby(df_sales['Date']).sum()
    # df_sales_group = df_sales.groupby(['Date'])[["TransactionCount"]].sum()

    df_smoothed_sales = pd.read_csv('data/sales_with_smoothing.csv')
    df_smoothed_sales = df_smoothed_sales[['Date', 'StoreId', 'SmoothedTransactionCount']]
    df_smoothed_sales['Date'] = pd.to_datetime(df_smoothed_sales.Date, yearfirst=True)

    df_activity_score = pd.read_csv('data/social_media_activity_scores.csv')
    df_activity_score = df_activity_score[['Date', 'Social Medial Activity Score', 'Total Impressions']]
    df_activity_score['Date'] = pd.to_datetime(df_activity_score.Date, format="%m-%d-%Y")
    df_activity_score['Total Impressions'] = df_activity_score['Total Impressions'].str.replace(',', '')
    df_activity_score['Total Impressions'] = pd.to_numeric(df_activity_score['Total Impressions'])

    # df_twitter = pd.read_csv('data/Twitter Profiles-01-01-2018-02-09-2020.csv')
    # df_twitter = df_twitter[['Date', 'Impressions', 'Published Posts']]
    # df_twitter['Date'] = pd.to_datetime(df_twitter.Date, format="%m-%d-%Y")
    # df_twitter.plot(grid=True)
    # plt.show()


    # Correlation of Two Time Series - Total transactions on a given date
    i = 1
    header = ['Stores',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    matrixtable = [header]
    corrpcttable = [header]
    matrixrowdf = pd.DataFrame()
    correlationrow = ['All']
    corrpctrow = ['All']
    while i <= 30:
        df_smoothed_sales['Date'] += datetime.timedelta(days=i)
        df_smoothed_sales_group = df_smoothed_sales.groupby(['Date'])[["SmoothedTransactionCount"]].sum()
        df = df_activity_score.join(df_smoothed_sales_group, on='Date', how='inner')
        newrow = find_correlation(i,'All',df, 'Social Medial Activity Score', 'SmoothedTransactionCount')
        # newrow = find_correlation(i, 'All', df, 'Total Impressions', 'SmoothedTransactionCount')
        # matrixrowdf = matrixrowdf.append(newrow, ignore_index=True)
        correlationrow.append(newrow['Correlation'])
        corrpctrow.append(newrow['CorrelationPercent'])
        i += 1
    matrixtable.append(correlationrow)
    corrpcttable.append(corrpctrow)
    # Get the unique values of 'B' column
    stores = df_smoothed_sales.StoreId.unique()
    # Using for loop
    for storeid in stores:
        df_filtered_sales = df_smoothed_sales[df_smoothed_sales['StoreId'] == storeid]
        # print(df_filtered_sales.dtypes)
        # print(df_activity_score.dtypes)
        i = 1
        # matrixrowdf = pd.DataFrame()
        correlationrow = [storeid]
        corrpctrow = [storeid]
        while i <= 30:
            df_filtered_sales['Date'] += datetime.timedelta(days=i)
            #df = df_activity_score.join(df_filtered_sales, on='Date', how='inner')
            df = pd.merge(df_filtered_sales, df_activity_score, on='Date', how='inner')
            newrow = find_correlation(i, storeid, df, 'Social Medial Activity Score', 'SmoothedTransactionCount')
            # newrow = find_correlation(i, storeid, df, 'Total Impressions', 'SmoothedTransactionCount')
            # matrixrowdf = matrixrowdf.append(newrow, ignore_index=True)
            correlationrow.append(newrow['Correlation'])
            corrpctrow.append(newrow['CorrelationPercent'])
            i += 1
        matrixtable.append(correlationrow)
        corrpcttable.append(corrpctrow)
    with open("data/correlation_heatmap.csv", "w+") as correlation_heatmap:
        csvWriter = csv.writer(correlation_heatmap, delimiter=',')
        csvWriter.writerows(matrixtable)
    with open("data/correlation_percent_heatmap.csv", "w+") as correlation_percent_heatmap:
        csvWriter = csv.writer(correlation_percent_heatmap, delimiter=',')
        csvWriter.writerows(matrixtable)
    print('CSV File, correlation_heatmap.csv, for correlation heatmap created')
    print('CSV File, correlation_percent_heatmap, for correlation percent heatmap created')
    # df = df_twitter.join(df_sales_group, on='Date', how='inner')
    # df.Impressions = df['Impressions'].str.replace(',', '')
    # df.Impressions = pd.to_numeric(df['Impressions'])
    # df.TransactionCount = pd.to_numeric(df['TransactionCount'])
    # df.index=df['Date']
    # print(df.info())

    # reggression = scipy.stats.linregress(df['Published Posts'], df['Impressions'])
    # print("Standard error is ", reggression.stderr)

    # find_correlation(df, 'Impressions', 'TransactionCount');
    # find_correlation(df, 'Published Posts', 'Impressions');
    # find_autocorrelation(df, 'Impressions')


def get_row_data(data, level, column_name, ID, target_social_media_score):
    store_data = data[data[column_name] == ID]
    data_without_score = store_data[[
        'Date',
        column_name,
        'SmoothedTransactionCount'
    ]]
    data_with_just_score = store_data[[
        'Date',
        target_social_media_score

    ]]
    data_without_score = data_without_score.groupby(['Date', column_name], as_index=False).sum()

    store_data = pd.merge(
        data_without_score,
        data_with_just_score,
        how='left',
        on='Date'
    )

    weighted_scores = store_data[target_social_media_score]
    smoothed_transactions = store_data['SmoothedTransactionCount']

    row = {'Level': level, 'ID': ID}

    number_of_days = 30

    for offset in range(number_of_days, -1, -1):
        smoothed_transactions = smoothed_transactions.shift(periods=-1, fill_value=np.nan)
        smoothed_transactions = smoothed_transactions.dropna()
        correlation = weighted_scores.corr(smoothed_transactions, method='pearson')
        row[str(number_of_days - offset)] = correlation

    return row


def determine_and_write_correlations(stores, combined, target_name, target_value):
    rows = []

    regions = stores['Region'].unique()
    markets = stores['Market'].unique()
    states = stores['State'].unique()
    cities = stores['City'].unique()
    store_ids = stores['StoreId'].unique()

    for region in regions:
        rows.append(get_row_data(combined, 'Region', 'Region', region, target_value))

    for market in markets:
        rows.append(get_row_data(combined, 'Market', 'Market', market, target_value))

    for state in states:
        rows.append(get_row_data(combined, 'State', 'State', state, target_value))

    for city in cities:
        rows.append(get_row_data(combined, 'City', 'City', city, target_value))

    for store_id in store_ids:
        rows.append(get_row_data(combined, 'Store', 'StoreId', store_id, target_value))

    df = pd.DataFrame(rows)

    df.to_csv('correlations_between_media({})_and_transactions.csv'.format(target_name), index=False)


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

    target_list = {
        'Posts': 'Social Medial Activity Score',
        'Weighted Posts': 'Weighted Social Medial Activity Score',
        'Engagement Scores': 'Engagement Scores',
        'Awareness': 'Awareness'
    }

    for target_name, target_value in target_list.items():
        determine_and_write_correlations(stores, combined, target_name, target_value)



if __name__ == '__main__':
    correlation_analysis()