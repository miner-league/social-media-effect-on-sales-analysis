import data_access
import feature_engineering
import smoothing
import aggregate

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
        self.data['stores'] = data_access.get_store_data()

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

    def calculate_sales_smoothing(self):
        self.data['sales_with_smoothing'] = smoothing.smooth_transaction_data()

    def combine_sales_with_loyalty(self):
        self.data['sales_with_loyalty'] = aggregate.combine_sales_with_loyalty(
            self.data['sales_with_smoothing'],
            self.data['loyalty'],
            self.data['stores']
        )
        self.data['sales_with_loyalty'].to_csv('data/sales_with_loyalty.csv', index=False)

    def aggregate_and_combine_data(self):
        self.combine_sales_with_loyalty()
