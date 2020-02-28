from data_access_object import DataAccessObject

def main():
    data_access_object = DataAccessObject()

    print('loading data...')
    data_access_object.load_data()

    print('calculating social media activity score...')
    data_access_object.calculate_social_media_score()
    data_access_object.data['social_media_activity_score'].to_csv('data/social_media_activity_scores.csv', index=False)

    print('smoothing sales...')
    data_access_object.calculate_sales_smoothing()
    data_access_object.data['sales_with_smoothing'].to_csv('data/sales_with_smoothing.csv', index=False)

    print('combining datasets for dependent variables...')
    data_access_object.aggregate_and_combine_data()

if __name__ == '__main__':
    main()
