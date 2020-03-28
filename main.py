from correlations import calculate_social_media_score_sales_correlations, \
    calculate_social_media_score_loyalty_correlations
from data_access_object import DataAccessObject
from twitter_data_imputation import impute_missing_twitter_data


def main():
    data_access_object = DataAccessObject()

    print('Impute missing Twitter data...')
    impute_missing_twitter_data()

    print('loading data...')
    data_access_object.load_data()
    #
    print('calculating social media activity score...')
    data_access_object.calculate_social_media_score()
    data_access_object.data['social_media_activity_score'].to_csv('data/social_media_activity_scores.csv', index=False)

    print('smoothing sales...')
    data_access_object.calculate_sales_smoothing()
    data_access_object.data['sales_with_smoothing'].to_csv('data/sales_with_smoothing.csv', index=False)

    print('combining datasets for dependent variables...')
    data_access_object.aggregate_and_combine_data()
    #
    print('calculating correlations for social media score and sales by store')
    calculate_social_media_score_sales_correlations()

    print('calculating correlations for social media score and loyalty')
    calculate_social_media_score_loyalty_correlations()

if __name__ == '__main__':
    main()
