import csv
import datetime
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from utilities import remove_commas, remove_percent
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


def get_twitter_social_media_data():
    df = pd.read_csv('data/twitter_imputed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts'] = pd.to_numeric(df['Published Posts'])
    df['Followers'] = df['Followers'].apply(remove_commas)
    df['Followers'] = pd.to_numeric(df['Followers'])
    df['Net Follower Growth'] = df['Net Follower Growth'].apply(remove_commas)
    df['Net Follower Growth'] = pd.to_numeric(df['Net Follower Growth'])
    df['Following'] = df['Following'].apply(remove_commas)
    df['Following'] = pd.to_numeric(df['Following'])
    df['Net Following Growth'] = df['Net Following Growth'].apply(remove_commas)
    df['Net Following Growth'] = pd.to_numeric(df['Net Following Growth'])
    df['Impressions'] = df['Impressions'].apply(remove_commas)
    df['Impressions'] = pd.to_numeric(df['Impressions'])
    df['Video Views'] = df['Video Views'].apply(remove_commas)
    df['Video Views'] = pd.to_numeric(df['Video Views'])
    df['Retweets'] = df['Retweets'].apply(remove_commas)
    df['Retweets'] = pd.to_numeric(df['Retweets'])
    df['Engagements'] = df['Engagements'].apply(remove_commas)
    df['Engagements'] = pd.to_numeric(df['Engagements'])
    df['Likes'] = df['Likes'].apply(remove_commas)
    df['Likes'] = pd.to_numeric(df['Likes'])
    df['Other Post Clicks'] = df['Other Post Clicks'].apply(remove_commas)
    df['Other Post Clicks'] = pd.to_numeric(df['Other Post Clicks'])
    df['Engagement Rate (per Impression)'] = df['Engagement Rate (per Impression)'].apply(remove_percent)
    df['Engagement Rate (per Impression)'] = pd.to_numeric(df['Engagement Rate (per Impression)'])
    return df


def get_facebook_social_media_data():
    df = pd.read_csv('data/Facebook Pages-01-01-2018-02-09-2020.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts'] = pd.to_numeric(df['Published Posts'])
    df['Total Fans'] = df['Total Fans'].apply(remove_commas)
    df['Total Fans'] = pd.to_numeric(df['Total Fans'])
    df['Impressions'] = df['Impressions'].apply(remove_commas)
    df['Impressions'] = pd.to_numeric(df['Impressions'])
    df['Organic Impressions'] = df['Organic Impressions'].apply(remove_commas)
    df['Organic Impressions'] = pd.to_numeric(df['Organic Impressions'])
    df['Paid Impressions'] = df['Paid Impressions'].apply(remove_commas)
    df['Paid Impressions'] = pd.to_numeric(df['Paid Impressions'])
    df['Reach'] = df['Reach'].apply(remove_commas)
    df['Reach'] = pd.to_numeric(df['Reach'])
    df['Organic Reach'] = df['Organic Reach'].apply(remove_commas)
    df['Organic Reach'] = pd.to_numeric(df['Organic Reach'])
    df['Paid Reach'] = df['Paid Reach'].apply(remove_commas)
    df['Paid Reach'] = pd.to_numeric(df['Paid Reach'])
    df['Engagements'] = df['Engagements'].apply(remove_commas)
    df['Engagements'] = pd.to_numeric(df['Engagements'])
    df['Post Link Clicks'] = df['Post Link Clicks'].apply(remove_commas)
    df['Post Link Clicks'] = pd.to_numeric(df['Post Link Clicks'])
    df['Other Post Clicks'] = df['Other Post Clicks'].apply(remove_commas)
    df['Other Post Clicks'] = pd.to_numeric(df['Other Post Clicks'])
    df['Partial Video Views'] = df['Partial Video Views'].apply(remove_commas)
    df['Partial Video Views'] = pd.to_numeric(df['Partial Video Views'])
    df['Reactions'] = df['Reactions'].apply(remove_commas)
    df['Reactions'] = pd.to_numeric(df['Reactions'])
    df['Likes'] = df['Likes'].apply(remove_commas)
    df['Likes'] = pd.to_numeric(df['Likes'])
    df['Video Views'] = df['Video Views'].apply(remove_commas)
    df['Video Views'] = pd.to_numeric(df['Video Views'])
    df['Full Video Views'] = df['Full Video Views'].apply(remove_commas)
    df['Full Video Views'] = pd.to_numeric(df['Full Video Views'])
    df['Organic Video Views'] = df['Organic Video Views'].apply(remove_commas)
    df['Organic Video Views'] = pd.to_numeric(df['Organic Video Views'])
    df['Organic Full Video Views'] = df['Organic Full Video Views'].apply(remove_commas)
    df['Organic Full Video Views'] = pd.to_numeric(df['Organic Full Video Views'])
    df['Organic Partial Video Views'] = df['Organic Partial Video Views'].apply(remove_commas)
    df['Organic Partial Video Views'] = pd.to_numeric(df['Organic Partial Video Views'])
    df['Paid Video Views'] = df['Paid Video Views'].apply(remove_commas)
    df['Paid Video Views'] = pd.to_numeric(df['Paid Video Views'])
    df['Paid Full Video Views'] = df['Paid Full Video Views'].apply(remove_commas)
    df['Paid Full Video Views'] = pd.to_numeric(df['Paid Full Video Views'])
    df['Paid Partial Video Views'] = df['Paid Partial Video Views'].apply(remove_commas)
    df['Paid Partial Video Views'] = pd.to_numeric(df['Paid Partial Video Views'])
    df['Click to Play Video Views'] = df['Click to Play Video Views'].apply(remove_commas)
    df['Click to Play Video Views'] = pd.to_numeric(df['Click to Play Video Views'])
    df['Autoplay Video Views'] = df['Autoplay Video Views'].apply(remove_commas)
    df['Autoplay Video Views'] = pd.to_numeric(df['Autoplay Video Views'])
    df['Engagement Rate (per Impression)'] = df['Engagement Rate (per Impression)'].apply(remove_percent)
    df['Engagement Rate (per Impression)'] = pd.to_numeric(df['Engagement Rate (per Impression)'])
    return df


def get_instagram_social_media_data():
    df = pd.read_csv('data/Instagram Business Profiles-01-01-2018-02-09-2020.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts & Stories'] = pd.to_numeric(df['Published Posts & Stories'])
    df['Followers'] = df['Followers'].apply(remove_commas)
    df['Followers'] = pd.to_numeric(df['Followers'])
    df['Impressions'] = df['Impressions'].apply(remove_commas)
    df['Impressions'] = pd.to_numeric(df['Impressions'])
    df['Reach'] = df['Reach'].apply(remove_commas)
    df['Reach'] = pd.to_numeric(df['Reach'])
    df['Engagements'] = df['Engagements'].apply(remove_commas)
    df['Engagements'] = pd.to_numeric(df['Engagements'])
    df['Engagement Rate (per Impression)'] = df['Engagement Rate (per Impression)'].apply(remove_percent)
    df['Engagement Rate (per Impression)'] = pd.to_numeric(df['Engagement Rate (per Impression)'])
    return df


def replace_missing_value(df, number_features):
    imputer = SimpleImputer(strategy="median")
    df_num = df[number_features]
    imputer.fit(df_num)
    X = imputer.transform(df_num)
    res_def = pd.DataFrame(X, columns=df_num.columns)
    return res_def


def get_combined_social_data():
    twitter_data = get_twitter_social_media_data()
    twitter_columns = ['Date', 'Twitter Profile', 'Twitter_Followers', 'Twitter_Net Follower Growth',
                       'Twitter_Following', 'Twitter_Net Following Growth', 'Twitter_Published Posts',
                       'Twitter_Impressions',
                       'Twitter_Video Views', 'Twitter_Engagements', 'Twitter_Likes', 'Twitter_@Replies',
                       'Twitter_Retweets',
                       'Twitter_Post Link Clicks', 'Twitter_Other Post Clicks', 'Twitter_Other Engagements',
                       'Twitter_Engagement Rate (per Impression)']
    twitter_data.columns = twitter_columns
    # print(twitter_data.columns)
    # print(twitter_columns)
    facebook_data = get_facebook_social_media_data()
    facebook_columns = ['Date', 'Facebook Page', 'Facebook_Total Fans', 'Facebook_Net Page Likes',
                        'Facebook_Page Likes',
                        'Facebook_Organic Page Likes', 'Facebook_Paid Page Likes', 'Facebook_Page Unlikes',
                        'Facebook_Published Posts', 'Facebook_Impressions', 'Facebook_Organic Impressions',
                        'Facebook_Paid Impressions', 'Facebook_Reach', 'Facebook_Organic Reach', 'Facebook_Paid Reach',
                        'Facebook_Engagements', 'Facebook_Reactions', 'Facebook_Likes', 'Facebook_Love Reactions',
                        'Facebook_Haha Reactions',
                        'Facebook_Wow Reactions', 'Facebook_Sad Reactions', 'Facebook_Angry Reactions',
                        'Facebook_Comments',
                        'Facebook_Shares', 'Facebook_Post Link Clicks', 'Facebook_Other Post Clicks',
                        'Facebook_Page Actions',
                        'Facebook_Engagement Rate (per Impression)', 'Facebook_Negative Feedback',
                        'Facebook_Video Views',
                        'Facebook_Full Video Views', 'Facebook_Partial Video Views', 'Facebook_Organic Video Views',
                        'Facebook_Organic Full Video Views', 'Facebook_Organic Partial Video Views',
                        'Facebook_Paid Video Views', 'Facebook_Paid Full Video Views',
                        'Facebook_Paid Partial Video Views',
                        'Facebook_Click to Play Video Views', 'Facebook_Autoplay Video Views']
    facebook_data.columns = facebook_columns
    # print(facebook_data.columns)
    # print(facebook_columns)
    instagram_data = get_instagram_social_media_data()
    instagram_columns = ['Date', 'Instagram Profile', 'Instagram_Followers', 'Instagram_Net Follower Growth',
                         'Instagram_Followers Gained', 'Instagram_Followers Lost', 'Instagram_Following',
                         'Instagram_Net Following Growth', 'Instagram_Published Posts & Stories',
                         'Instagram_Impressions',
                         'Instagram_Reach', 'Instagram_Engagements', 'Instagram_Likes', 'Instagram_Comments',
                         'Instagram_Saves', 'Instagram_Story Replies', 'Instagram_Profile Actions',
                         'Instagram_Engagement Rate (per Impression)']
    instagram_data.columns = instagram_columns
    # print(instagram_data.columns)
    # print(instagram_columns)

    tf = pd.merge(twitter_data, facebook_data, on='Date')
    tfi = pd.merge(tf, instagram_data, on='Date')
    return tfi


def generate_heatmap_transactions():
    tfi = get_combined_social_data()
    x_columns = ['Twitter_Followers',
                 'Twitter_Net Follower Growth', 'Twitter_Following',
                 'Twitter_Net Following Growth', 'Twitter_Published Posts',
                 'Twitter_Impressions', 'Twitter_Video Views', 'Twitter_Engagements',
                 'Twitter_Likes', 'Twitter_@Replies', 'Twitter_Retweets',
                 'Twitter_Post Link Clicks', 'Twitter_Other Post Clicks',
                 'Twitter_Other Engagements', 'Twitter_Engagement Rate (per Impression)',
                 'Facebook_Total Fans', 'Facebook_Net Page Likes',
                 'Facebook_Page Likes', 'Facebook_Organic Page Likes',
                 'Facebook_Paid Page Likes', 'Facebook_Page Unlikes',
                 'Facebook_Published Posts', 'Facebook_Impressions',
                 'Facebook_Organic Impressions', 'Facebook_Paid Impressions',
                 'Facebook_Reach', 'Facebook_Organic Reach', 'Facebook_Paid Reach',
                 'Facebook_Engagements', 'Facebook_Reactions', 'Facebook_Likes',
                 'Facebook_Love Reactions', 'Facebook_Haha Reactions',
                 'Facebook_Wow Reactions', 'Facebook_Sad Reactions',
                 'Facebook_Angry Reactions', 'Facebook_Comments', 'Facebook_Shares',
                 'Facebook_Post Link Clicks', 'Facebook_Other Post Clicks',
                 'Facebook_Page Actions', 'Facebook_Engagement Rate (per Impression)',
                 'Facebook_Negative Feedback', 'Facebook_Video Views',
                 'Facebook_Full Video Views', 'Facebook_Partial Video Views',
                 'Facebook_Organic Video Views', 'Facebook_Organic Full Video Views',
                 'Facebook_Organic Partial Video Views', 'Facebook_Paid Video Views',
                 'Facebook_Paid Full Video Views', 'Facebook_Paid Partial Video Views',
                 'Facebook_Click to Play Video Views', 'Facebook_Autoplay Video Views',
                 'Instagram_Followers',
                 'Instagram_Net Follower Growth', 'Instagram_Followers Gained',
                 'Instagram_Followers Lost', 'Instagram_Following',
                 'Instagram_Net Following Growth', 'Instagram_Published Posts & Stories',
                 'Instagram_Impressions', 'Instagram_Reach', 'Instagram_Engagements',
                 'Instagram_Likes', 'Instagram_Comments', 'Instagram_Saves',
                 'Instagram_Story Replies', 'Instagram_Profile Actions',
                 'Instagram_Engagement Rate (per Impression)']
    df_smoothed_sales = pd.read_csv('data/sales_with_smoothing.csv')
    df_smoothed_sales = df_smoothed_sales[['Date', 'StoreId', 'SmoothedTransactionCount']]
    df_smoothed_sales['Date'] = pd.to_datetime(df_smoothed_sales.Date, yearfirst=True)
    i = 1
    header = ['Stores', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
              27, 28, 29, 30]
    matrixtable = [header]
    rsquared_adj_row = ['All']
    while i <= 30:
        df_smoothed_sales['Date'] += datetime.timedelta(days=i)
        df_smoothed_sales_group = df_smoothed_sales.groupby(['Date'])[["SmoothedTransactionCount"]].sum()
        tfi_smooth_trans = pd.merge(tfi, df_smoothed_sales_group, on='Date')
        X = tfi_smooth_trans[x_columns]
        X = replace_missing_value(X, X.columns)
        Y = tfi_smooth_trans['SmoothedTransactionCount']
        regressor = LinearRegression()
        regressor.fit(X, Y)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        pred_p_values = model.pvalues[model.pvalues < 0.05]
        pred_x_columns = pred_p_values.keys().tolist()
        if 'const' in pred_x_columns:
            pred_x_columns.remove('const')

        # Model with predominant features
        rsquared_adj = 0
        if len(pred_x_columns) >= 1:
            X2 = tfi_smooth_trans[pred_x_columns]
            X2 = replace_missing_value(X2, X2.columns)
            regressor2 = LinearRegression()
            regressor2.fit(X2, Y)
            X2 = sm.add_constant(X2)
            model2 = sm.OLS(Y, X2).fit()
            rsquared_adj = model2.rsquared_adj
        rsquared_adj_row.append(rsquared_adj)
        i += 1
    matrixtable.append(rsquared_adj_row)
    # Get the unique values of 'B' column
    stores = df_smoothed_sales.StoreId.unique()
    # Using for loop
    for storeid in stores:
        df_filtered_sales = df_smoothed_sales[df_smoothed_sales['StoreId'] == storeid]
        i = 1
        rsquared_adj_row = [storeid]
        while i <= 30:
            df_filtered_sales['Date'] += datetime.timedelta(days=i)
            df_smoothed_sales_group = df_filtered_sales.groupby(['Date'])[["SmoothedTransactionCount"]].sum()
            tfi_smooth_trans = pd.merge(tfi, df_smoothed_sales_group, on='Date')
            X = tfi_smooth_trans[x_columns]
            X = replace_missing_value(X, X.columns)
            Y = tfi_smooth_trans['SmoothedTransactionCount']
            regressor = LinearRegression()
            regressor.fit(X, Y)
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            pred_p_values = model.pvalues[model.pvalues < 0.05]
            pred_x_columns = pred_p_values.keys().tolist()
            if 'const' in pred_x_columns:
                pred_x_columns.remove('const')

            # Model with predominant features
            rsquared_adj = 0
            if len(pred_x_columns) >= 1:
                X2 = tfi_smooth_trans[pred_x_columns]
                X2 = replace_missing_value(X2, X2.columns)
                regressor2 = LinearRegression()
                regressor2.fit(X2, Y)
                X2 = sm.add_constant(X2)
                model2 = sm.OLS(Y, X2).fit()
                rsquared_adj = model2.rsquared_adj
            rsquared_adj_row.append(rsquared_adj)
            i += 1
        matrixtable.append(rsquared_adj_row)
    with open("data/multiple_lr_heatmap_transactions.csv", "w+") as multiple_lr_heatmap:
        csvWriter = csv.writer(multiple_lr_heatmap, delimiter=',')
        csvWriter.writerows(matrixtable)


def generate_mlfit_loyalty():
    tfi = get_combined_social_data()
    tfi['week'] = tfi['Date'].dt.week
    tfi['year'] = tfi['Date'].dt.year
    tfi = tfi.drop(columns=['Date'])
    tfi_aggregated = tfi.groupby(['year', 'week'], as_index=False).sum()
    x_columns = ['Twitter_Followers',
                 'Twitter_Net Follower Growth', 'Twitter_Following',
                 'Twitter_Net Following Growth', 'Twitter_Published Posts',
                 'Twitter_Impressions', 'Twitter_Video Views', 'Twitter_Engagements',
                 'Twitter_Likes', 'Twitter_@Replies', 'Twitter_Retweets',
                 'Twitter_Post Link Clicks', 'Twitter_Other Post Clicks',
                 'Twitter_Other Engagements', 'Twitter_Engagement Rate (per Impression)',
                 'Facebook_Total Fans', 'Facebook_Net Page Likes',
                 'Facebook_Page Likes', 'Facebook_Organic Page Likes',
                 'Facebook_Paid Page Likes', 'Facebook_Page Unlikes',
                 'Facebook_Published Posts', 'Facebook_Impressions',
                 'Facebook_Organic Impressions', 'Facebook_Paid Impressions',
                 'Facebook_Reach', 'Facebook_Organic Reach', 'Facebook_Paid Reach',
                 'Facebook_Engagements', 'Facebook_Reactions', 'Facebook_Likes',
                 'Facebook_Love Reactions', 'Facebook_Haha Reactions',
                 'Facebook_Wow Reactions', 'Facebook_Sad Reactions',
                 'Facebook_Angry Reactions', 'Facebook_Comments', 'Facebook_Shares',
                 'Facebook_Post Link Clicks', 'Facebook_Other Post Clicks',
                 'Facebook_Page Actions', 'Facebook_Engagement Rate (per Impression)',
                 'Facebook_Negative Feedback', 'Facebook_Video Views',
                 'Facebook_Full Video Views', 'Facebook_Partial Video Views',
                 'Facebook_Organic Video Views', 'Facebook_Organic Full Video Views',
                 'Facebook_Organic Partial Video Views', 'Facebook_Paid Video Views',
                 'Facebook_Paid Full Video Views', 'Facebook_Paid Partial Video Views',
                 'Facebook_Click to Play Video Views', 'Facebook_Autoplay Video Views',
                 'Instagram_Followers',
                 'Instagram_Net Follower Growth', 'Instagram_Followers Gained',
                 'Instagram_Followers Lost', 'Instagram_Following',
                 'Instagram_Net Following Growth', 'Instagram_Published Posts & Stories',
                 'Instagram_Impressions', 'Instagram_Reach', 'Instagram_Engagements',
                 'Instagram_Likes', 'Instagram_Comments', 'Instagram_Saves',
                 'Instagram_Story Replies', 'Instagram_Profile Actions',
                 'Instagram_Engagement Rate (per Impression)']
    loyalty_scores = pd.read_excel('data/loyalty.xlsx')
    loyalty_scores['Date'] = pd.to_datetime(loyalty_scores['CalendarWeekEndingDate'])
    loyalty_scores['week'] = loyalty_scores['Date'].dt.week
    loyalty_scores['year'] = loyalty_scores['Date'].dt.year
    loyalty_scores = loyalty_scores.drop(columns=['Date', 'CalendarWeekEndingDate'])
    loyalty_scores_aggregated = loyalty_scores.groupby(['year', 'week', 'State'], as_index=False).sum()
    i = 1
    header = ['State', 'rsquared_adj', 'RMSE']
    matrixtable = [header]
    rows = ['All']
    tfi_loyalty = pd.merge(
        tfi_aggregated,
        loyalty_scores_aggregated,
        how='inner',
        on=['year', 'week']
    )
    tfi_loyalty = tfi_loyalty.dropna()
    X = tfi_loyalty[x_columns]
    X = replace_missing_value(X, X.columns)
    Y = tfi_loyalty['NewRegistrations']
    regressor = LinearRegression()
    regressor.fit(X, Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    pred_p_values = model.pvalues[model.pvalues < 0.05]
    pred_x_columns = pred_p_values.keys().tolist()
    if 'const' in pred_x_columns:
        pred_x_columns.remove('const')

    # Model with predominant features
    rsquared_adj = 0
    if len(pred_x_columns) >= 1:
        X2 = tfi_loyalty[pred_x_columns]
        X2 = replace_missing_value(X2, X2.columns)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X2, Y, random_state=1)
        regressor2 = LinearRegression()
        regressor2.fit(X_train, y_train)
        X_train2 = sm.add_constant(X_train)
        model2 = sm.OLS(y_train, X_train2).fit()
        rsquared_adj = model2.rsquared_adj
        # Predict
        y_pred = regressor2.predict(X_test)
        # RMSE
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rows.append(rsquared_adj)
        rows.append(rmse)
    matrixtable.append(rows)

    states = loyalty_scores_aggregated['State'].unique()
    for state in states:
        loyalty_scores_aggregated_state = loyalty_scores_aggregated[loyalty_scores_aggregated['State'] == state]
        if len(loyalty_scores_aggregated_state) >= 1:
            tfi_loyalty_state = pd.merge(
                tfi_aggregated,
                loyalty_scores_aggregated_state,
                how='inner',
                on=['year', 'week']
            )
            rows = [state]
            tfi_loyalty_state = tfi_loyalty_state.dropna()
            X = tfi_loyalty_state[x_columns]
            X = replace_missing_value(X, X.columns)
            Y = tfi_loyalty_state['NewRegistrations']
            regressor = LinearRegression()
            regressor.fit(X, Y)
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            pred_p_values = model.pvalues[model.pvalues < 0.05]
            pred_x_columns = pred_p_values.keys().tolist()
            if 'const' in pred_x_columns:
                pred_x_columns.remove('const')

            # Model with predominant features
            rsquared_adj = 0
            if len(pred_x_columns) >= 1:
                X2 = tfi_loyalty_state[pred_x_columns]
                X2 = replace_missing_value(X2, X2.columns)
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X2, Y, random_state=1)
                regressor2 = LinearRegression()
                regressor2.fit(X_train, y_train)
                X_train2 = sm.add_constant(X_train)
                model2 = sm.OLS(y_train, X_train2).fit()
                rsquared_adj = model2.rsquared_adj
                # Predict
                y_pred = regressor2.predict(X_test)
                # RMSE
                rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                rows.append(rsquared_adj)
                rows.append(rmse)
            matrixtable.append(rows)

    with open("data/generate_mlfit_loyalty.csv", "w+") as generate_mlfit_loyalty:
        csvWriter = csv.writer(generate_mlfit_loyalty, delimiter=',')
        csvWriter.writerows(matrixtable)


x_columns = ['Twitter_Followers',
              'Twitter_Net Follower Growth', 'Twitter_Following',
              'Twitter_Net Following Growth', 'Twitter_Published Posts',
              'Twitter_Impressions', 'Twitter_Video Views', 'Twitter_Engagements',
              'Twitter_Likes', 'Twitter_@Replies', 'Twitter_Retweets',
              'Twitter_Post Link Clicks', 'Twitter_Other Post Clicks',
              'Twitter_Other Engagements', 'Twitter_Engagement Rate (per Impression)',
              'Facebook_Total Fans', 'Facebook_Net Page Likes',
              'Facebook_Page Likes', 'Facebook_Organic Page Likes',
              'Facebook_Paid Page Likes', 'Facebook_Page Unlikes',
              'Facebook_Published Posts', 'Facebook_Impressions',
              'Facebook_Organic Impressions', 'Facebook_Paid Impressions',
              'Facebook_Reach', 'Facebook_Organic Reach', 'Facebook_Paid Reach',
              'Facebook_Engagements', 'Facebook_Reactions', 'Facebook_Likes',
              'Facebook_Love Reactions', 'Facebook_Haha Reactions',
              'Facebook_Wow Reactions', 'Facebook_Sad Reactions',
              'Facebook_Angry Reactions', 'Facebook_Comments', 'Facebook_Shares',
              'Facebook_Post Link Clicks', 'Facebook_Other Post Clicks',
              'Facebook_Page Actions', 'Facebook_Engagement Rate (per Impression)',
              'Facebook_Negative Feedback', 'Facebook_Video Views',
              'Facebook_Full Video Views', 'Facebook_Partial Video Views',
              'Facebook_Organic Video Views', 'Facebook_Organic Full Video Views',
              'Facebook_Organic Partial Video Views', 'Facebook_Paid Video Views',
              'Facebook_Paid Full Video Views', 'Facebook_Paid Partial Video Views',
              'Facebook_Click to Play Video Views', 'Facebook_Autoplay Video Views',
              'Instagram_Followers',
              'Instagram_Net Follower Growth', 'Instagram_Followers Gained',
              'Instagram_Followers Lost', 'Instagram_Following',
              'Instagram_Net Following Growth', 'Instagram_Published Posts & Stories',
              'Instagram_Impressions', 'Instagram_Reach', 'Instagram_Engagements',
              'Instagram_Likes', 'Instagram_Comments', 'Instagram_Saves',
              'Instagram_Story Replies', 'Instagram_Profile Actions',
              'Instagram_Engagement Rate (per Impression)']


def generate_mlfit_loyalty1():
    x_columns = ['ActivityScore', 'WeightedActivityScore', 'TotalImpressions']
    scores_df = pd.read_csv('data/social_media_activity_scores.csv')
    scores_df['Date'] = pd.to_datetime(scores_df['Date'])
    scores_df['ActivityScore'] = scores_df['Social Medial Activity Score'].apply(remove_commas)
    scores_df['ActivityScore'] = pd.to_numeric(scores_df['ActivityScore'])
    scores_df['WeightedActivityScore'] = scores_df['Weighted Social Medial Activity Score'].apply(remove_commas)
    scores_df['WeightedActivityScore'] = pd.to_numeric(scores_df['WeightedActivityScore'])
    scores_df['TotalImpressions'] = scores_df['Total Impressions'].apply(remove_commas)
    scores_df['TotalImpressions'] = pd.to_numeric(scores_df['TotalImpressions'])
    tfi = scores_df[['Date', 'ActivityScore', 'WeightedActivityScore', 'TotalImpressions']]
    # tfi = get_combined_social_data()
    tfi['week'] = tfi['Date'].dt.week
    tfi['year'] = tfi['Date'].dt.year
    tfi = tfi.drop(columns=['Date'])
    tfi_aggregated = tfi.groupby(['year', 'week'], as_index=False).sum()

    loyalty_scores = pd.read_excel('data/loyalty.xlsx')
    loyalty_scores['Date'] = pd.to_datetime(loyalty_scores['CalendarWeekEndingDate'])
    loyalty_scores['week'] = loyalty_scores['Date'].dt.week
    loyalty_scores['year'] = loyalty_scores['Date'].dt.year
    loyalty_scores = loyalty_scores.drop(columns=['Date', 'CalendarWeekEndingDate'])
    loyalty_scores_aggregated = loyalty_scores.groupby(['year', 'week', 'State'], as_index=False).sum()
    i = 1
    header = ['State', 'rsquared_adj', 'RMSE']
    matrixtable = [header]
    tfi_loyalty = pd.merge(
        tfi_aggregated,
        loyalty_scores_aggregated,
        how='inner',
        on=['year', 'week']
    )
    min = tfi_loyalty['NewRegistrations'].min()
    max = tfi_loyalty['NewRegistrations'].max()
    rows = ['All: '+str(min)+' - '+str(max)]
    matrixtable = addMatrixTableRow(x_columns, matrixtable, rows, tfi_loyalty, tfi_loyalty['NewRegistrations'])

    states = loyalty_scores_aggregated['State'].unique()
    for state in states:
        loyalty_scores_aggregated_state = loyalty_scores_aggregated[loyalty_scores_aggregated['State'] == state]
        if len(loyalty_scores_aggregated_state) >= 1:
            tfi_loyalty_state = pd.merge(
                tfi_aggregated,
                loyalty_scores_aggregated_state,
                how='inner',
                on=['year', 'week']
            )
            rows = [state]
            matrixtable = addMatrixTableRow(x_columns, matrixtable, rows, tfi_loyalty_state,
                                            tfi_loyalty_state['NewRegistrations'])

    with open("data/generate_mlfit_loyalty1.csv", "w+") as generate_mlfit_loyalty:
        csvWriter = csv.writer(generate_mlfit_loyalty, delimiter=',')
        csvWriter.writerows(matrixtable)


def get_combined_sales_data(aggregationType):
    sales_df = pd.read_csv('data/sales_with_smoothing.csv')
    stores_df = pd.read_excel('data/stores.xlsx')
    result_df = pd.merge(sales_df, stores_df[['StoreId', 'City', 'State', 'Region']], on='StoreId')
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    result_df['month'] = result_df['Date'].dt.month
    result_df['year'] = result_df['Date'].dt.year
    result_df_aggregated = ''
    if aggregationType == 'State':
        result_df = result_df.drop(columns=['Date', 'StoreId', 'City', 'Region'])
        result_df_aggregated = result_df.groupby(['year', 'month', 'State'], as_index=False).sum()
    elif aggregationType == 'Region':
        result_df = result_df.drop(columns=['Date', 'StoreId', 'City', 'State'])
        result_df_aggregated = result_df.groupby(['year', 'month', 'Region'], as_index=False).sum()
    elif aggregationType == 'City':
        result_df = result_df.drop(columns=['Date', 'StoreId', 'State', 'Region'])
        result_df_aggregated = result_df.groupby(['year', 'month', 'City'], as_index=False).sum()
    elif aggregationType == 'Store':
        result_df = result_df.drop(columns=['Date', 'City', 'State', 'Region'])
        result_df_aggregated = result_df.groupby(['year', 'month', 'StoreId'], as_index=False).sum()
    return result_df_aggregated


def generate_mlfit_sales():
    x_columns = ['ActivityScore', 'WeightedActivityScore', 'TotalImpressions']
    scores_df = pd.read_csv('data/social_media_activity_scores.csv')
    scores_df['Date'] = pd.to_datetime(scores_df['Date'])
    scores_df['ActivityScore'] = scores_df['Social Medial Activity Score'].apply(remove_commas)
    scores_df['ActivityScore'] = pd.to_numeric(scores_df['ActivityScore'])
    scores_df['WeightedActivityScore'] = scores_df['Weighted Social Medial Activity Score'].apply(remove_commas)
    scores_df['WeightedActivityScore'] = pd.to_numeric(scores_df['WeightedActivityScore'])
    scores_df['TotalImpressions'] = scores_df['Total Impressions'].apply(remove_commas)
    scores_df['TotalImpressions'] = pd.to_numeric(scores_df['TotalImpressions'])
    tfi = scores_df[['Date', 'ActivityScore', 'WeightedActivityScore', 'TotalImpressions']]

    # tfi = get_combined_social_data()
    tfi['month'] = tfi['Date'].dt.month
    tfi['year'] = tfi['Date'].dt.year
    tfi = tfi.drop(columns=['Date'])
    tfi_aggregated = tfi.groupby(['year', 'month'], as_index=False).sum()

    ##### Store based ML analysis #####
    sales_df_store = get_combined_sales_data('Store')
    matrixtable_store = [['Store', 'R Squared_adj', 'RMSE']]
    all_store_sales = pd.merge(
        tfi_aggregated,
        sales_df_store,
        how='inner',
        on=['year', 'month']
    )
    min=all_store_sales['SmoothedTransactionCount'].min()
    max = all_store_sales['SmoothedTransactionCount'].max()
    all_store_sales = all_store_sales.groupby(['year', 'month'], as_index=False).sum()
    matrixtable_store = addMatrixTableRow(x_columns, matrixtable_store, ['All: '+str(min)+' - '+str(max)], all_store_sales,
                                          all_store_sales['SmoothedTransactionCount'])
    stores = sales_df_store['StoreId'].unique()
    for storeId in stores:
        sales_df_aggregated_store = sales_df_store[sales_df_store['StoreId'] == storeId]
        if len(sales_df_aggregated_store) >= 1:
            store_sales = pd.merge(
                tfi_aggregated,
                sales_df_aggregated_store,
                how='inner',
                on=['year', 'month']
            )
            store_sales = store_sales.groupby(['year', 'month'], as_index=False).sum()
            matrixtable_store = addMatrixTableRow(x_columns, matrixtable_store, [storeId], store_sales,
                                                  store_sales['SmoothedTransactionCount'])
    with open("data/sales_mlfit_store.csv", "w+") as sales_mlfit_store:
        csvWriter = csv.writer(sales_mlfit_store, delimiter=',')
        csvWriter.writerows(matrixtable_store)

    ##### City based ML analysis #####
    sales_df_city = get_combined_sales_data('City')
    matrixtable_city = [['City', 'R Squared_adj', 'RMSE']]
    all_city_sales = pd.merge(
        tfi_aggregated,
        sales_df_city,
        how='inner',
        on=['year', 'month']
    )
    min = all_city_sales['SmoothedTransactionCount'].min()
    max = all_city_sales['SmoothedTransactionCount'].max()
    all_city_sales = all_city_sales.groupby(['year', 'month'], as_index=False).sum()
    matrixtable_city = addMatrixTableRow(x_columns, matrixtable_city, ['All: '+str(min)+' - '+str(max)], all_city_sales,
                                         all_city_sales['SmoothedTransactionCount'])
    cities = sales_df_city['City'].unique()
    for city in cities:
        sales_df_aggregated_city = sales_df_city[sales_df_city['City'] == city]
        if len(sales_df_aggregated_city) >= 1:
            city_sales = pd.merge(
                tfi_aggregated,
                sales_df_aggregated_city,
                how='inner',
                on=['year', 'month']
            )
            city_sales = city_sales.groupby(['year', 'month'], as_index=False).sum()
            matrixtable_city = addMatrixTableRow(x_columns, matrixtable_city, [city], city_sales,
                                                 city_sales['SmoothedTransactionCount'])
    with open("data/sales_mlfit_city.csv", "w+") as sales_mlfit_city:
        csvWriter = csv.writer(sales_mlfit_city, delimiter=',')
        csvWriter.writerows(matrixtable_city)

    ##### State based ML analysis #####
    sales_df_state = get_combined_sales_data('State')
    matrixtable_state = [['State', 'R Squared_adj', 'RMSE']]
    all_state_sales = pd.merge(
        tfi_aggregated,
        sales_df_state,
        how='inner',
        on=['year', 'month']
    )
    min = all_state_sales['SmoothedTransactionCount'].min()
    max = all_state_sales['SmoothedTransactionCount'].max()
    all_state_sales = all_state_sales.groupby(['year', 'month'], as_index=False).sum()
    matrixtable_state = addMatrixTableRow(x_columns, matrixtable_state, ['All: '+str(min)+' - '+str(max)], all_state_sales,
                                          all_state_sales['SmoothedTransactionCount'])
    states = sales_df_state['State'].unique()
    for state in states:
        sales_df_aggregated_state = sales_df_state[sales_df_state['State'] == state]
        if len(sales_df_aggregated_state) >= 1:
            state_sales = pd.merge(
                tfi_aggregated,
                sales_df_aggregated_state,
                how='inner',
                on=['year', 'month']
            )
            matrixtable_state = addMatrixTableRow(x_columns, matrixtable_state, [state], state_sales,
                                                  state_sales['SmoothedTransactionCount'])
    with open("data/sales_mlfit_state.csv", "w+") as sales_mlfit_state:
        csvWriter = csv.writer(sales_mlfit_state, delimiter=',')
        csvWriter.writerows(matrixtable_state)

    ##### Region based ML analysis #####
    sales_df_region = get_combined_sales_data('Region')
    matrixtable_region = [['Region', 'R Squared_adj', 'RMSE']]
    all_region_sales = pd.merge(
        tfi_aggregated,
        sales_df_region,
        how='inner',
        on=['year', 'month']
    )
    min = all_region_sales['SmoothedTransactionCount'].min()
    max = all_region_sales['SmoothedTransactionCount'].max()
    matrixtable_region = addMatrixTableRow(x_columns, matrixtable_region, ['All: '+str(min)+' - '+str(max)], all_region_sales,
                                           all_region_sales['SmoothedTransactionCount'])
    regions = sales_df_region['Region'].unique()
    for region in regions:
        sales_df_aggregated_region = sales_df_region[sales_df_region['Region'] == region]
        if len(sales_df_aggregated_region) >= 1:
            region_sales = pd.merge(
                tfi_aggregated,
                sales_df_aggregated_region,
                how='inner',
                on=['year', 'month']
            )
            matrixtable_region = addMatrixTableRow(x_columns, matrixtable_region, [region], region_sales,
                                                   region_sales['SmoothedTransactionCount'])
    with open("data/sales_mlfit_region.csv", "w+") as sales_mlfit_region:
        csvWriter = csv.writer(sales_mlfit_region, delimiter=',')
        csvWriter.writerows(matrixtable_region)


def addMatrixTableRow(x_columns, matrixtable, rows, df, Y):
    df = df.dropna()
    X = df[x_columns]
    X = replace_missing_value(X, X.columns)
    # Y = df['SmoothedTransactionCount']

    # regressor = LinearRegression()
    # regressor.fit(X, Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    pred_p_values = model.pvalues[model.pvalues < 0.05]
    pred_x_columns = pred_p_values.keys().tolist()
    if 'const' in pred_x_columns:
        pred_x_columns.remove('const')

    # Model with predominant features
    rsquared_adj = 0
    if len(pred_x_columns) >= 1:
        X2 = df[pred_x_columns]
        X2 = replace_missing_value(X2, X2.columns)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X2, Y, random_state=1)
        regressor2 = LinearRegression()
        regressor2.fit(X_train, y_train)
        X_train2 = sm.add_constant(X_train)
        model2 = sm.OLS(y_train, X_train2).fit()
        rsquared_adj = model2.rsquared_adj
        # Predict
        y_pred = regressor2.predict(X_test)
        # RMSE
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rows.append(rsquared_adj)
        rows.append(rmse)
    matrixtable.append(rows)
    return matrixtable


def pca_analysis():
    tfi = get_combined_social_data()
    tfi['month'] = tfi['Date'].dt.month
    tfi['year'] = tfi['Date'].dt.year
    tfi = tfi.drop(columns=['Date'])
    tfi_aggregated = tfi.groupby(['year', 'month'], as_index=False).sum()

    ##### Store based ML analysis #####
    sales_df_store = get_combined_sales_data('Store')
    all_store_sales = pd.merge(
        tfi_aggregated,
        sales_df_store,
        how='inner',
        on=['year', 'month']
    )
    all_store_sales = all_store_sales.groupby(['year', 'month'], as_index=False).sum()
    all_store_sales = all_store_sales.dropna()
    X = all_store_sales[x_columns]
    X = replace_missing_value(X, X.columns)
    Y = all_store_sales['SmoothedTransactionCount']
    pca_process_xy(X, Y)


def pca_process_xy(x, y):

    # Splitting the X and Y into the
    # Training set and Testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # performing preprocessing part
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Applying PCA function on training
    # and testing set of X component
    from sklearn.decomposition import PCA

    pca = PCA(n_components = 2)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_

    # Fitting Logistic Regression To the training set
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the test set result using
    # predict function under LogisticRegression
    y_pred = classifier.predict(X_test)

    # making confusion matrix between
    # test set of Y and predicted value.
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    # Predicting the training set
    # result through scatter plot
    from matplotlib.colors import ListedColormap

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                        stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1,
                        stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
                cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

    plt.title('Logistic Regression (Training set)')
    plt.xlabel('PC1') # for Xlabel
    plt.ylabel('PC2') # for Ylabel
    plt.legend() # to show legend

    # show scatter plot
    plt.show()

    # Visualising the Test set results through scatter plot
    from matplotlib.colors import ListedColormap

    X_set, y_set = X_test, y_test

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                        stop = X_set[:, 0].max() + 1, step = 0.01),
                        np.arange(start = X_set[:, 1].min() - 1,
                        stop = X_set[:, 1].max() + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
                cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

    # title for scatter plot
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('PC1') # for Xlabel
    plt.ylabel('PC2') # for Ylabel
    plt.legend()

    # show scatter plot
    plt.show()

def generate_mlfit_nps():
    x_columns = ['ActivityScore', 'WeightedActivityScore', 'TotalImpressions']
    scores_df = pd.read_csv('data/social_media_activity_scores.csv')
    scores_df['Date'] = pd.to_datetime(scores_df['Date'])
    scores_df['ActivityScore'] = scores_df['Social Medial Activity Score'].apply(remove_commas)
    scores_df['ActivityScore'] = pd.to_numeric(scores_df['ActivityScore'])
    scores_df['WeightedActivityScore'] = scores_df['Weighted Social Medial Activity Score'].apply(remove_commas)
    scores_df['WeightedActivityScore'] = pd.to_numeric(scores_df['WeightedActivityScore'])
    scores_df['TotalImpressions'] = scores_df['Total Impressions'].apply(remove_commas)
    scores_df['TotalImpressions'] = pd.to_numeric(scores_df['TotalImpressions'])
    tfi = scores_df[['Date', 'ActivityScore', 'WeightedActivityScore', 'TotalImpressions']]
    tfi['Month'] = tfi['Date'].dt.month
    tfi['Year'] = tfi['Date'].dt.year
    tfi = tfi.drop(columns=['Date'])
    tfi_aggregated = tfi.groupby(['Year', 'Month'], as_index=False).sum()

    nps_scores = pd.read_excel('data/nps.xlsx')
    nps_scores_aggregated = nps_scores.groupby(['Year', 'Month','StoreId'], as_index=False).sum()
    header = ['Store', 'rsquared_adj', 'RMSE']
    matrixtable = [header]
    tfi_nps = pd.merge(
        tfi_aggregated,
        nps_scores_aggregated,
        on=['Year', 'Month']
    )
    tfi_nps = tfi_nps.dropna()
    min = tfi_nps['NPSScore'].min()
    max = tfi_nps['NPSScore'].max()
    rows = ['All: '+str(min)+' - '+str(max)]
    X = tfi_nps[x_columns]
    X = replace_missing_value(X, X.columns)
    Y = tfi_nps['NPSScore']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    pred_p_values = model.pvalues[model.pvalues < 0.05]
    pred_x_columns = pred_p_values.keys().tolist()
    if 'const' in pred_x_columns:
        pred_x_columns.remove('const')

    # Model with predominant features
    rsquared_adj = 0
    if len(pred_x_columns) >= 1:
        X2 = tfi_nps[pred_x_columns]
        X2 = replace_missing_value(X2, X2.columns)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X2, Y, random_state=1)
        regressor2 = LinearRegression()
        regressor2.fit(X_train, y_train)
        X_train2 = sm.add_constant(X_train)
        model2 = sm.OLS(y_train, X_train2).fit()
        rsquared_adj = model2.rsquared_adj
        # Predict
        y_pred = regressor2.predict(X_test)
        # RMSE
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        rows.append(rsquared_adj)
        rows.append(rmse)
    matrixtable.append(rows)
    stores = nps_scores.StoreId.unique()
    for store in stores:
        nps_scores_aggregated_store = nps_scores_aggregated[nps_scores_aggregated['StoreId'] == store]
        if len(nps_scores_aggregated_store) >= 1:
            tfi_nps_store = pd.merge(
                tfi_aggregated,
                nps_scores_aggregated_store,
                on=['Year', 'Month']
            )
            rows = [store]
            X = tfi_nps_store[x_columns]
            X = replace_missing_value(X, X.columns)
            Y = tfi_nps_store['NPSScore']
            regressor = LinearRegression()
            regressor.fit(X, Y)
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            pred_p_values = model.pvalues[model.pvalues < 0.05]
            pred_x_columns = pred_p_values.keys().tolist()
            if 'const' in pred_x_columns:
                pred_x_columns.remove('const')

            # Model with predominant features
            rsquared_adj = 0
            if len(pred_x_columns) >= 1:
                X2 = tfi_nps_store[pred_x_columns]
                X2 = replace_missing_value(X2, X2.columns)
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X2, Y, random_state=1)
                regressor2 = LinearRegression()
                regressor2.fit(X_train, y_train)
                X_train2 = sm.add_constant(X_train)
                model2 = sm.OLS(y_train, X_train2).fit()
                rsquared_adj = model2.rsquared_adj
                # Predict
                y_pred = regressor2.predict(X_test)
                # RMSE
                rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                rows.append(rsquared_adj)
                rows.append(rmse)
            matrixtable.append(rows)

    with open("data/generate_mlfit_nps.csv", "w+") as generate_mlfit_nps:
        csvWriter = csv.writer(generate_mlfit_nps, delimiter=',')
        csvWriter.writerows(matrixtable)

if __name__ == '__main__':
    # generate_heatmap_transactions()
    # generate_mlfit_loyalty()
    # generate_mlfit_loyalty1()
    # generate_mlfit_sales()
    # pca_analysis()
    generate_mlfit_nps()
