import datetime
import json

import luigi
import pandas as pd
from sklearn import ensemble

from docs.conf import *
from love_matcher.refactored.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher.refactored.preprocessing.raw_set_processing import RawSetProcessing
from love_matcher.refactored.split_test_train import SplitTestTrain
from love_matcher.refactored.training.training import Trainer
from love_matcher.refactored.tuning.tuning import TuneParameters




class FeatureEngineeringTask:
    def run(self):
        dataset = self.read_dataframe()

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=dataset)

        # Feature engineering
        feature_engineering = FeatureEngineering()
        feat_eng_df, processed_features_names = feature_engineering.get_partner_features(dataset_df)
        feat_eng_df.to_csv(feature_engineered_dataset_file_path)

    def read_dataframe(self):
        return pd.read_csv(workspace + "Speed_Dating_Data.csv", encoding="ISO-8859-1")


class TuneTask:
    def run(self):
        # TODO make random forest parameters externalized
        feat_eng_df = pd.read_csv(feature_engineered_dataset_file_path)
        rf_model = ensemble.RandomForestClassifier(n_estimators=5, class_weight="balanced", oob_score=False)
        tune = TuneParameters(feat_eng_df, rf_model, parameters, scores, features)
        best_parameters = tune.combiner_pipeline()
        self.save_best_parameters(best_parameters)

    def save_best_parameters(self, best_parameters):
        with open(best_parameters_file_path, 'w') as best_parameters_file:
            json.dump(best_parameters, best_parameters_file)
            best_parameters_file.close()


class TrainTask:
    def run(self):
        feat_eng_df = pd.read_csv(feature_engineered_dataset_file_path)
        with open(best_parameters_file_path) as best_parameters_file:
            best_parameters = json.load(best_parameters_file)
            best_parameters_file.close()
            split_test_train = SplitTestTrain(feat_eng_df=feat_eng_df)
            x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()
            train = Trainer(x_train, y_train, x_test, y_test, best_parameters)
            estimator, score_train, score_test = train.combiner_pipeline()
            print(estimator, score_train, score_test)
