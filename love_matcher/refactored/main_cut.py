import pandas as pd
from sklearn import ensemble

from love_matcher.refactored.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher.refactored.preprocessing.raw_set_processing import RawSetProcessing
from love_matcher.refactored.training.training import Trainer
from love_matcher.refactored.tuning.tuning import TuneParameters
from docs.conf import *


class MainClass:
    # TODO make workspace configurable from luigi.cfg
    def __init__(self, workspace="/Users/sandrapietrowska/Documents/Trainings/luigi/data_source/"):
        self.workspace = workspace

    def main(self):
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
        best_parameters = tune.combiner_pipeline()

        x_train, x_test, y_train, y_test = tune.create_train_test_splits()

        # Train
        train = Trainer(x_train, y_train, x_test, y_test, best_parameters)
        estimator, score_train, score_test = train.combiner_pipeline()
        print (estimator, score_train, score_test)

    def read_dataframe(self):
        return pd.read_csv(self.workspace + "Speed_Dating_Data.csv", encoding="ISO-8859-1")


if __name__ == '__main__':
    MainClass().main()


