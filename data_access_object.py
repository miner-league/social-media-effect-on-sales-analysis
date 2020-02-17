import data_access
import feature_engineering


class DataAccessObject:

    def __init__(self):
        self.data = {}

    def get_facebook_social_media_data(self):
        self.data['facebook_social_media'] = data_access.get_facebook_social_media_data()

    def get_twitter_social_media_data(self):
        self.data['twitter_social_media'] = data_access.get_twitter_social_media_data()

    def get_instagram_social_media_data(self):
        self.data['instagram_social_media'] = data_access.get_instagram_social_media_data()

    def get_loyalty_data(self):
        self.data['loyalty'] = data_access.get_loyalty_data()

    def get_survey_data(self):
        self.data['survey'] = data_access.get_survey_data()

    def get_sales_data(self):
        self.data['sales'] = data_access.get_sales_data()

    def get_store_data(self):
        self.data['store'] = data_access.get_store_data()

    def load_data(self):
        self.get_facebook_social_media_data()
        self.get_twitter_social_media_data()
        self.get_instagram_social_media_data()
        self.get_loyalty_data()
        self.get_survey_data()
        self.get_sales_data()
        self.get_store_data()

    def calculate_social_media_score(self):
        self.data['social_media_activity_score'] = feature_engineering.determine_social_media_scores(self.data)
