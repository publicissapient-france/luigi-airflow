import datetime
import json

import os
import pandas as pd
from sklearn import ensemble

from docs.conf import *
from love_matcher.refactored.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher.refactored.preprocessing.raw_set_processing import RawSetProcessing
from love_matcher.refactored.training.training import Trainer
from love_matcher.refactored.tuning.tuning import TuneParameters


class MainClass:
    # TODO make workspace configurable from luigi.cfg
    def __init__(self, workspace="/home/dolounet/dev/workshops/"):
        self.workspace = workspace

    def main(self):
        start_time = datetime.datetime.now()
        dataset = self.read_dataframe()

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=dataset)

        # Feature engineering
        feature_engineering = FeatureEngineering()
        feat_eng_df = feature_engineering.get_partner_features(dataset_df)

        # TODO make random forest parameters externalized

        rf_model = ensemble.RandomForestClassifier(n_estimators=5, class_weight="balanced", oob_score=False)

        tune = TuneParameters(feat_eng_df, rf_model, parameters, scores, features)
        if not os.path.exists(best_parameters_file_path):
            best_parameters = tune.combiner_pipeline()
            self.save_best_parameters(best_parameters)

        best_parameters_loaded = self.load_best_parameters()

        # Train
        x_train, x_test, y_train, y_test = tune.create_train_test_splits()
        train = Trainer(x_train, y_train, x_test, y_test, best_parameters_loaded)
        estimator, score_train, score_test = train.combiner_pipeline()
        print(estimator, score_train, score_test)
        end_time = datetime.datetime.now()

    def save_best_parameters(self, best_parameters):
        with open(best_parameters_file_path, 'w') as best_parameters_file:
            json.dump(best_parameters, best_parameters_file)
            best_parameters_file.close()

    def load_best_parameters(self):
        with open(best_parameters_file_path) as best_parameters_file:
            return json.load(best_parameters_file)

    def read_dataframe(self):
        return pd.read_csv(self.workspace + "Speed_Dating_Data.csv", encoding="ISO-8859-1")


if __name__ == '__main__':
    MainClass().main()
