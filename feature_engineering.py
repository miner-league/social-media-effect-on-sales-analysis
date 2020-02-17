import pandas as pd
from datetime import date, timedelta


def date_range(first_date, last_date):
    for number in range(int((last_date - first_date).days) + 1):
        yield first_date + timedelta(number)


def get_dates():
    # Social media data is availale from 1-1-2018 to 2-9-2020.  In future versions this could be derived.
    first_date = date(2018, 1, 1)
    last_date = date(2020, 2, 9)

    dates = []
    for dt in date_range(first_date, last_date):
        dates.append(dt.strftime('%m-%d-%Y'))

    return dates


def get_facebook_score(row):
    # Facebook raw score is just number of published posts
    post_count = row['Published Posts'].values[0]

    # Multiply by number of Total Fans to 'weight' the count
    number_of_fans = row['Total Fans'].values[0]

    return post_count, post_count * number_of_fans


def get_twitter_score(row):
    # Twitter raw score is just number of published posts
    post_count = row['Published Posts'].values[0]

    # Multiply by number of followers to 'weight' the count
    number_of_followers = row['Followers'].values[0]

    return post_count, post_count * number_of_followers


def get_instagram_score(row):
    # Instagram raw score is also published posts and stories
    post_count = row['Published Posts & Stories'].values[0]

    # Multiply by number of followers to 'weight' the count
    number_of_followers = row['Followers'].values[0]

    return post_count, post_count * number_of_followers


def get_relevant_row(data, dt, media_type):
    facebook_data = data[media_type]
    relevant_row = facebook_data.loc[facebook_data['Date'] == dt]
    return relevant_row


def get_scores(dates, data):
    raw_scores = []
    weighted_scores = []
    for dt in dates:
        raw_facebook_score, weighted_facebook_score = get_facebook_score(get_relevant_row(data, dt, 'facebook_social_media'))
        raw_twitter_score, weighted_twitter_score = get_twitter_score(get_relevant_row(data, dt, 'twitter_social_media'))
        raw_instagram_score, weighted_instagram_score = get_instagram_score(get_relevant_row(data, dt, 'instagram_social_media'))

        raw_score = raw_facebook_score + raw_twitter_score + raw_instagram_score
        raw_scores.append(raw_score)

        weighted_score = weighted_facebook_score + weighted_twitter_score + weighted_instagram_score
        weighted_scores.append(weighted_score)

    return raw_scores, weighted_scores


def determine_social_media_scores(data):
    dates = get_dates()
    raw_scores, weighted_scores = get_scores(dates, data)

    return pd.DataFrame({
        'Date': dates,
        'Social Medial Activity Score': raw_scores,
        'Weighted Social Medial Activity Score': weighted_scores
    })