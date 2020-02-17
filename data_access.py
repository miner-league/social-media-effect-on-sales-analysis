import pandas as pd
from utilities import remove_commas


def get_facebook_social_media_data():
    df = pd.read_csv('data/Facebook Pages-01-01-2018-02-09-2020.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts'] = pd.to_numeric(df['Published Posts'])
    df['Total Fans'] = df['Total Fans'].apply(remove_commas)
    df['Total Fans'] = pd.to_numeric(df['Total Fans'])
    return df


def get_twitter_social_media_data():
    df = pd.read_csv('data/Twitter Profiles-01-01-2018-02-09-2020.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts'] = pd.to_numeric(df['Published Posts'])
    df['Followers'] = df['Followers'].apply(remove_commas)
    df['Followers'] = pd.to_numeric(df['Followers'])
    return df


def get_instagram_social_media_data():
    df = pd.read_csv('data/Instagram Business Profiles-01-01-2018-02-09-2020.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts & Stories'] = pd.to_numeric(df['Published Posts & Stories'])
    df['Followers'] = df['Followers'].apply(remove_commas)
    df['Followers'] = pd.to_numeric(df['Followers'])
    return df


def get_loyalty_data():
    return pd.read_excel('data/loyalty.xlsx')


def get_survey_data():
    return pd.read_excel('data/nps.xlsx')


def get_sales_data():
    return pd.read_excel('data/sales.xlsx')


def get_store_data():
    return pd.read_excel('data/stores.xlsx')
