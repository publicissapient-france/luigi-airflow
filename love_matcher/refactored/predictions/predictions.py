import pandas as pd
from love_matcher.refactored.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher.refactored.preprocessing.raw_set_processing import RawSetProcessing
import warnings
from docs.conf import *
from sklearn.externals import joblib

class Predictor:
    warnings.filterwarnings("ignore")

    def __init__(self, new_data, x_train, y_train, model_type):
        self.new_data = new_data
        self.x_train = x_train
        self.y_train = y_train
        self.model_type = model_type

    def load_estimator(self, model_target):
        loaded_estimator = joblib.load(model_target + '/' + self.model_type + '_model.pkl')
        return loaded_estimator

    def predict(self, estimator):
        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=self.new_data)

        # Feature engineering
        feature_engineering = FeatureEngineering(features=features)
        all_features_engineered_df, selected_features_df = feature_engineering.get_partner_features(dataset_df)
        reg = self.load_estimator(model_target=output_dir)
        predictions_labels = reg.predict(selected_features_df)
        return predictions_labels

    def export_pred_to_csv(self, preds):
        pred_df = pd.DataFrame(preds)
        pred_concat_df = pd.concat([pred_df,self.new_data], axis=1)
        pred_concat_df.to_csv(output_dir + "/" + self.model_type + "_predictions.csv",index=False)