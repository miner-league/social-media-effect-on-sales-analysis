
import pandas as pd
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
    df['Engagement Rate (per Impression)'] = df['Engagement Rate (per Impression)'].apply(remove_percent)
    df['Engagement Rate (per Impression)'] = pd.to_numeric(df['Engagement Rate (per Impression)'])
    return df

def get_facebook_social_media_data():
    df = pd.read_csv('data/Facebook Pages-01-01-2018-02-09-2020.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts'] = pd.to_numeric(df['Published Posts'])
    df['Total Fans'] = df['Total Fans'].apply(remove_commas)
    df['Total Fans'] = pd.to_numeric(df['Total Fans'])
    df['Engagement Rate (per Impression)'] = df['Engagement Rate (per Impression)'].apply(remove_percent)
    df['Engagement Rate (per Impression)'] = pd.to_numeric(df['Engagement Rate (per Impression)'])
    return df

def get_instagram_social_media_data():
    df = pd.read_csv('data/Instagram Business Profiles-01-01-2018-02-09-2020.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Published Posts & Stories'] = pd.to_numeric(df['Published Posts & Stories'])
    df['Followers'] = df['Followers'].apply(remove_commas)
    df['Followers'] = pd.to_numeric(df['Followers'])
    df['Engagement Rate (per Impression)'] = df['Engagement Rate (per Impression)'].apply(remove_percent)
    df['Engagement Rate (per Impression)'] = pd.to_numeric(df['Engagement Rate (per Impression)'])
    return df

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
    tfi_smooth_trans.to_csv('data/social_media_data_smooth_trans.csv', index=False)


if __name__ == '__main__':
    process_data()



