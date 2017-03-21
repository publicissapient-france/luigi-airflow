import pandas as pd
from sklearn import ensemble

from love_matcher.refactored.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher.refactored.preprocessing.raw_set_processing import RawSetProcessing
from love_matcher.refactored.training.training import Trainer
from love_matcher.refactored.tuning.tuning import TuneParameters
from docs.conf import *


class MainClass:
    # TODO make workspace configurable from luigi.cfg
    def __init__(self, workspace="/home/dolounet/dev/workshops/"):
        self.workspace = workspace

    def main(self):
        raw_dataset = self.read_dataframe()

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection, dataframe=raw_dataset)
        dataset_df = raw_dataset.combiner_pipeline()

        suffix_me = "_me"
        suffix_partner = "_partner"

        my_label = "match_perc"
        label = "match"

        # Feature engineering
        feature_engineering = FeatureEngineering(suffix_1=suffix_me, suffix_2=suffix_partner, label=my_label)
        feat_eng_df = feature_engineering.get_partner_features(dataset_df)

        features_model = self.process_features_names(features, suffix_me, suffix_partner)

        # TODO make random forest parameters externalized
        rf_model = ensemble.RandomForestClassifier(n_estimators=5, oob_score=False)

        # Tuning
        explanatory = feat_eng_df[features_model]
        explained = feat_eng_df[label]
        tune = TuneParameters(explanatory, explained, rf_model, parameters, scores)
        best_parameters = tune.combiner_pipeline()
        x_train, x_test, y_train, y_test = tune.create_train_test_splits()

        # Train
        train = Trainer(x_train, y_train, x_test, y_test, best_parameters)
        estimator, score_train, score_test = train.combiner_pipeline()

    def read_dataframe(self):
        return pd.read_csv(self.workspace + "Speed_Dating_Data.csv", encoding="ISO-8859-1")

    def process_features_names(self, features, suffix_1, suffix_2):
        features_me = [feat + suffix_1 for feat in features]
        features_partner = [feat + suffix_2 for feat in features]
        features_all = features_me + features_partner
        return features_all


# add suffix to each element of list


