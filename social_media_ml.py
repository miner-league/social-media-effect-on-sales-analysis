import csv
import datetime
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from utilities import remove_commas, remove_percent


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

def process_data():
    twitter_data = get_twitter_social_media_data()
    twitter_columns = ['Date', 'Twitter Profile', 'Twitter_Followers', 'Twitter_Net Follower Growth',
       'Twitter_Following', 'Twitter_Net Following Growth', 'Twitter_Published Posts', 'Twitter_Impressions',
       'Twitter_Video Views', 'Twitter_Engagements', 'Twitter_Likes', 'Twitter_@Replies', 'Twitter_Retweets',
       'Twitter_Post Link Clicks', 'Twitter_Other Post Clicks', 'Twitter_Other Engagements',
       'Twitter_Engagement Rate (per Impression)']
    twitter_data.columns = twitter_columns
    # print(twitter_data.columns)
    # print(twitter_columns)
    facebook_data = get_facebook_social_media_data()
    facebook_columns = ['Date', 'Facebook Page', 'Facebook_Total Fans', 'Facebook_Net Page Likes', 'Facebook_Page Likes',
       'Facebook_Organic Page Likes', 'Facebook_Paid Page Likes', 'Facebook_Page Unlikes',
       'Facebook_Published Posts', 'Facebook_Impressions', 'Facebook_Organic Impressions',
       'Facebook_Paid Impressions', 'Facebook_Reach', 'Facebook_Organic Reach', 'Facebook_Paid Reach',
       'Facebook_Engagements', 'Facebook_Reactions', 'Facebook_Likes', 'Facebook_Love Reactions', 'Facebook_Haha Reactions',
       'Facebook_Wow Reactions', 'Facebook_Sad Reactions', 'Facebook_Angry Reactions', 'Facebook_Comments',
       'Facebook_Shares', 'Facebook_Post Link Clicks', 'Facebook_Other Post Clicks', 'Facebook_Page Actions',
       'Facebook_Engagement Rate (per Impression)', 'Facebook_Negative Feedback', 'Facebook_Video Views',
       'Facebook_Full Video Views', 'Facebook_Partial Video Views', 'Facebook_Organic Video Views',
       'Facebook_Organic Full Video Views', 'Facebook_Organic Partial Video Views',
       'Facebook_Paid Video Views', 'Facebook_Paid Full Video Views', 'Facebook_Paid Partial Video Views',
       'Facebook_Click to Play Video Views', 'Facebook_Autoplay Video Views']
    facebook_data.columns = facebook_columns
    # print(facebook_data.columns)
    # print(facebook_columns)
    instagram_data = get_instagram_social_media_data()
    instagram_columns = ['Date', 'Instagram Profile', 'Instagram_Followers', 'Instagram_Net Follower Growth',
       'Instagram_Followers Gained', 'Instagram_Followers Lost', 'Instagram_Following',
       'Instagram_Net Following Growth', 'Instagram_Published Posts & Stories', 'Instagram_Impressions',
       'Instagram_Reach', 'Instagram_Engagements', 'Instagram_Likes', 'Instagram_Comments', 'Instagram_Saves', 'Instagram_Story Replies','Instagram_Profile Actions', 'Instagram_Engagement Rate (per Impression)']
    instagram_data.columns = instagram_columns
    # print(instagram_data.columns)
    # print(instagram_columns)

    tf = pd.merge(twitter_data, facebook_data, on='Date')
    tfi = pd.merge(tf, instagram_data, on='Date')
    # print(tfi.columns)
    df_smoothed_sales = pd.read_csv('data/sales_with_smoothing.csv')
    df_smoothed_sales = df_smoothed_sales[['Date', 'StoreId', 'SmoothedTransactionCount']]
    df_smoothed_sales['Date'] = pd.to_datetime(df_smoothed_sales.Date, yearfirst=True)
    df_smoothed_sales_group = df_smoothed_sales.groupby(['Date'])[["SmoothedTransactionCount"]].sum()
    tfi_smooth_trans = pd.merge(tfi, df_smoothed_sales_group, on='Date')
    # print(tfi_smooth_trans.columns)
    # tfi_smooth_trans.to_csv('data/social_media_data_smooth_trans.csv', index=False)
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
    X = tfi_smooth_trans[x_columns]
    X = replace_missing_value(X, X.columns)
    Y = tfi_smooth_trans['SmoothedTransactionCount']
    regressor = LinearRegression()
    regressor.fit(X, Y)
    # print('Intercept: \n', regressor.intercept_)
    # print('Coefficients: \n', regressor.coef_)
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    # predictions = model.predict(X)
    # print_model = model.summary().tables[1]
    # print(model.rsquared)
    # print(print_model)
    pred_p_values = model.pvalues[model.pvalues < 0.05]
    pred_x_columns = pred_p_values.keys().tolist()
    pred_x_columns.remove('const')
    # print(pred_x_columns)

    # Model with predominant features

    X2 = tfi_smooth_trans[pred_x_columns]
    X2 = replace_missing_value(X2, X2.columns)
    regressor2 = LinearRegression()
    regressor2.fit(X2, Y)
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(Y, X2).fit()
    # predictions2 = model2.predict(X2)
    print_model2 = model2.summary().tables[1]
    print(model2.rsquared_adj)
    print(print_model2)

def get_combined_social_data():
    twitter_data = get_twitter_social_media_data()
    twitter_columns = ['Date', 'Twitter Profile', 'Twitter_Followers', 'Twitter_Net Follower Growth',
       'Twitter_Following', 'Twitter_Net Following Growth', 'Twitter_Published Posts', 'Twitter_Impressions',
       'Twitter_Video Views', 'Twitter_Engagements', 'Twitter_Likes', 'Twitter_@Replies', 'Twitter_Retweets',
       'Twitter_Post Link Clicks', 'Twitter_Other Post Clicks', 'Twitter_Other Engagements',
       'Twitter_Engagement Rate (per Impression)']
    twitter_data.columns = twitter_columns
    # print(twitter_data.columns)
    # print(twitter_columns)
    facebook_data = get_facebook_social_media_data()
    facebook_columns = ['Date', 'Facebook Page', 'Facebook_Total Fans', 'Facebook_Net Page Likes', 'Facebook_Page Likes',
       'Facebook_Organic Page Likes', 'Facebook_Paid Page Likes', 'Facebook_Page Unlikes',
       'Facebook_Published Posts', 'Facebook_Impressions', 'Facebook_Organic Impressions',
       'Facebook_Paid Impressions', 'Facebook_Reach', 'Facebook_Organic Reach', 'Facebook_Paid Reach',
       'Facebook_Engagements', 'Facebook_Reactions', 'Facebook_Likes', 'Facebook_Love Reactions', 'Facebook_Haha Reactions',
       'Facebook_Wow Reactions', 'Facebook_Sad Reactions', 'Facebook_Angry Reactions', 'Facebook_Comments',
       'Facebook_Shares', 'Facebook_Post Link Clicks', 'Facebook_Other Post Clicks', 'Facebook_Page Actions',
       'Facebook_Engagement Rate (per Impression)', 'Facebook_Negative Feedback', 'Facebook_Video Views',
       'Facebook_Full Video Views', 'Facebook_Partial Video Views', 'Facebook_Organic Video Views',
       'Facebook_Organic Full Video Views', 'Facebook_Organic Partial Video Views',
       'Facebook_Paid Video Views', 'Facebook_Paid Full Video Views', 'Facebook_Paid Partial Video Views',
       'Facebook_Click to Play Video Views', 'Facebook_Autoplay Video Views']
    facebook_data.columns = facebook_columns
    # print(facebook_data.columns)
    # print(facebook_columns)
    instagram_data = get_instagram_social_media_data()
    instagram_columns = ['Date', 'Instagram Profile', 'Instagram_Followers', 'Instagram_Net Follower Growth',
       'Instagram_Followers Gained', 'Instagram_Followers Lost', 'Instagram_Following',
       'Instagram_Net Following Growth', 'Instagram_Published Posts & Stories', 'Instagram_Impressions',
       'Instagram_Reach', 'Instagram_Engagements', 'Instagram_Likes', 'Instagram_Comments', 'Instagram_Saves', 'Instagram_Story Replies','Instagram_Profile Actions', 'Instagram_Engagement Rate (per Impression)']
    instagram_data.columns = instagram_columns
    # print(instagram_data.columns)
    # print(instagram_columns)

    tf = pd.merge(twitter_data, facebook_data, on='Date')
    tfi = pd.merge(tf, instagram_data, on='Date')
    return tfi

def generate_heatmap():
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
    with open("data/multiple_lr_heatmap.csv", "w+") as multiple_lr_heatmap:
        csvWriter = csv.writer(multiple_lr_heatmap, delimiter=',')
        csvWriter.writerows(matrixtable)
if __name__ == '__main__':
    generate_heatmap()
    # process_data()



