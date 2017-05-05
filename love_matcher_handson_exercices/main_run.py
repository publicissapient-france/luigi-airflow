import datetime
import json

import os
import pandas as pd

from docs.conf import *
from love_matcher_handson_exercices.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher_handson_exercices.preprocessing.raw_data_preprocessing import RawSetProcessing
from love_matcher_handson_exercices.utils.split_train_test import SplitTestTrain
from love_matcher_handson_exercices.training.training import Trainer
from love_matcher_handson_exercices.evaluation.evaluation import Evaluator
from love_matcher_handson_exercices.predictions.predictions import Predictor

class MainClass:
    def __init__(self, model_type):
        self.model_type = model_type

    def main(self):
        start_time = datetime.datetime.now()
        dataset = self.read_dataframe("Speed_Dating_Data.csv")
        new_data = self.read_dataframe("Submission_set.csv", sep=";")

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=dataset)

        # Feature engineering
        feature_engineering = FeatureEngineering(features=features)
        all_features_engineered_df, selected_features_df = feature_engineering.get_partner_features(dataset_df)

        # Train
        split_test_train = SplitTestTrain(feat_eng_df=all_features_engineered_df, processed_features_names=selected_features_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()
        train = Trainer(x_train, y_train, x_test, y_test, model_type=self.model_type)
        train.save_estimator(output_dir)
        estimator, score_train, score_test = train.combiner_pipeline()
        end_time = datetime.datetime.now()
        print(end_time - start_time)

        # Evaluation
        evaluation = Evaluator(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, model_type = self.model_type)
        evaluation.eval()

        # Predictions
        predictions = Predictor(new_data=new_data, model_type=self.model_type)
        estimator = predictions.load_estimator(output_dir)
        predictions_applied = predictions.predict(estimator)
        predictions.export_pred_to_csv(predictions_applied)

    def save_best_parameters(self, best_parameters):
        print (self.model_type)
        with open(output_dir + "/" + self.model_type + "_best_parameters.json", 'w') as best_parameters_file:
            json.dump(best_parameters, best_parameters_file)
            best_parameters_file.close()

    def load_best_parameters(self):
        with open(output_dir + "/" + self.model_type + "_best_parameters.json") as best_parameters_file:
            return json.load(best_parameters_file)

    def read_dataframe(self, filename, sep=","):
        return pd.read_csv(workspace + filename, sep=sep, encoding="ISO-8859-1")


if __name__ == '__main__':
    MainClass(model_type="Decision_Tree").main()
