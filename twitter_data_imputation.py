import pandas as pd
from utilities import remove_commas, remove_percent


def get_twitter_social_media_data():
    df = pd.read_csv('data/Twitter Profiles-01-01-2018-02-09-2020.csv')
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

def main():
    twitter_data = get_twitter_social_media_data()
    twitter_data['Followers'].interpolate(method='linear', inplace=True, limit_direction="both")
    twitter_data['Followers'] = twitter_data['Followers'].round()
    twitter_data['Net Follower Growth'].interpolate(method='linear', inplace=True, limit_direction="both")
    twitter_data['Net Follower Growth'] = twitter_data['Net Follower Growth'].round()
    twitter_data['Following'].interpolate(method='linear', inplace=True, limit_direction="both")
    twitter_data['Following'] = twitter_data['Following'].round()
    twitter_data['Net Following Growth'].interpolate(method='linear', inplace=True, limit_direction="both")
    twitter_data['Net Following Growth'] = twitter_data['Net Following Growth'].round()
    twitter_data.to_csv('data/twitter_imputed.csv', index=False)

if __name__ == '__main__':
    main()
