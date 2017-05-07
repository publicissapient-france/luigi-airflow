import warnings

import pandas as pd
from sklearn.externals import joblib

from config.conf import *
from love_matcher_exercises.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher_exercises.preprocessing.raw_data_preprocessing import RawSetProcessing


class Predictor:
    warnings.filterwarnings("ignore")

    def __init__(self, new_data, model_type):
        self.new_data = new_data
        self.model_type = model_type

    def load_estimator(self, model_target):
        loaded_estimator = joblib.load(model_target + '/' + self.model_type + '_model.pkl')
        return loaded_estimator

    def predict(self, estimator):
        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection_pred)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=self.new_data)

        # Feature engineering
        feature_engineering = FeatureEngineering(features=features)
        all_features_engineered_df, selected_features_df = feature_engineering.add_partner_features_test(dataset_df)

        # Load model and generate predictions
        estimator = self.load_estimator(model_target=output_dir)
        print("Predictions...")
        # TODO: 5.1 Apply method predict on selected_features_df. Return predictions
        # TODO: Verify your solution by running test_predictions.py
        predictions_df = pd.DataFrame()  # Fill this constructor with your predictions array

        # Save predictions
        predictions_df.columns = ['Prediction_match']
        predictions_concat_df = pd.concat([predictions_df, all_features_engineered_df], axis=1)
        predictions_concat_df.to_csv(output_dir + "/" + self.model_type + "_predictions.csv", index=False)
        return predictions_concat_df
