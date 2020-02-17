from data_access_object import DataAccessObject

def main():
    data_access_object = DataAccessObject()

    data_access_object.load_data()
    data_access_object.calculate_social_media_score()

    data_access_object.data['social_media_activity_score'].to_csv('social_media_activity_scores.csv', index=False)

if __name__ == '__main__':
    main()
