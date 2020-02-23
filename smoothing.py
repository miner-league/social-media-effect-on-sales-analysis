from matplotlib import pyplot
import numpy as np
import pandas as pd


def build_indices(element_list, resolution):
    if resolution == 1: return [index for index in range(0, len(element_list))]
    else: return [index % resolution for index in range(0, len(element_list))]


def get_curve(degree, coefficients, indices):
    curve = []

    for i in range(len(indices)):
        value = coefficients[-1]
        for d in range(degree):
            value += indices[i] ** (degree - d) * coefficients[d]
        curve.append(value)

    return curve


def get_yearly_seasonal_curve(element_list, degree):
    indices = build_indices(element_list, 365)
    coefficients = np.polyfit(indices, element_list.values, degree)
    return get_curve(4, coefficients, indices)


def get_weekly_seasonal_curve(element_list, degree):
    indices = build_indices(element_list, 7)
    coefficients = np.polyfit(indices, element_list.values, degree)
    return get_curve(4, coefficients, indices)


def get_trend_line(element_list):
    indices = build_indices(element_list, 1)
    coefficients = np.polyfit(indices, element_list.values, 1)
    return get_curve(1, coefficients, indices)


def get_smoothed_transaction_data(store_id = 73):
    all_transactions = pd.read_excel('data/sales.xlsx')

    store_transactions = all_transactions.loc[all_transactions['StoreId'] == store_id]

    transactions = store_transactions['TransactionCount']
    mean = transactions.mean()
    # dates = store_transactions['Date']
    # pyplot.plot(dates, transactions, color='blue')

    yearly_seasonal_curve = get_yearly_seasonal_curve(transactions, 4)
    # pyplot.plot(dates, yearly_seasonal_curve, color='red')

    residuals = transactions - yearly_seasonal_curve
    # pyplot.plot(dates, residuals, color='blue')

    weekly_seasonal_curve = get_weekly_seasonal_curve(residuals, 4)
    # pyplot.plot(dates, weekly_seasonal_curve, color='red')

    residuals = residuals - weekly_seasonal_curve
    #pyplot.plot(dates, residuals, color='blue')

    trend_line = get_trend_line(residuals)
    # pyplot.plot(dates, trend_line, color='red')

    residuals = residuals - trend_line
    #pyplot.plot(dates, residuals, color='red')

    padded_residuals = residuals + mean
    rounded_padded_smoothed_transactions = padded_residuals.round()

    # pyplot.plot(dates, rounded_padded_smoothed_transactions, color='red')
    # pyplot.show()

    store_transactions['SmoothedTransactionCount'] = rounded_padded_smoothed_transactions

    return store_transactions


def get_store_ids():
    all_stores = pd.read_excel('data/stores.xlsx')
    return all_stores['StoreId']


def smooth_transaction_data():
    store_ids = get_store_ids()
    smoothed_df = pd.DataFrame()
    for store_id in store_ids:
        store_transactions = get_smoothed_transaction_data(store_id)
        smoothed_df = pd.concat([smoothed_df, store_transactions], sort=False)

    return smoothed_df

if __name__ == '__main__':
    get_smoothed_transaction_data()