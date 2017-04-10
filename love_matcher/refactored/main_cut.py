import datetime
import json

import os
import pandas as pd
from sklearn import ensemble

from docs.conf import *
from love_matcher.refactored.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher.refactored.preprocessing.raw_set_processing import RawSetProcessing
from love_matcher.refactored.split_test_train import SplitTestTrain
from love_matcher.refactored.training.training import Trainer
from love_matcher.refactored.tuning.tuning import TuneParameters
from love_matcher.refactored.evaluation.evaluation import Evaluator
from love_matcher.refactored.predictions.predictions import Predictor

class MainClass:
    def main(self):
        start_time = datetime.datetime.now()
        dataset = self.read_dataframe("Speed_Dating_Data.csv")
        new_data = self.read_dataframe("New_data.csv")
        model = "other"

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=dataset)

        # Feature engineering
        feature_engineering = FeatureEngineering(features=features)
        all_features_engineered_df, selected_features_df = feature_engineering.get_partner_features(dataset_df)
        print (all_features_engineered_df.shape)

        # TODO make random forest parameters externalized

        rf_model = ensemble.RandomForestClassifier(n_estimators=5, class_weight="balanced", oob_score=False)

        tune = TuneParameters(all_features_engineered_df, rf_model, parameters, scores, features)
        if not os.path.exists(best_parameters_file_path):
            best_parameters = tune.combiner_pipeline()[1]
            self.save_best_parameters(best_parameters)

        best_parameters_loaded = self.load_best_parameters()

        # Train
        split_test_train = SplitTestTrain(feat_eng_df=all_features_engineered_df, processed_features_names=selected_features_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()
        train = Trainer(x_train, y_train, x_test, y_test, best_parameters_loaded, model_type=model)
        train.save_estimator(output_dir)

        estimator, score_train, score_test = train.combiner_pipeline()
        end_time = datetime.datetime.now()
        print(end_time - start_time)

        # Evaluation
        evaluation = Evaluator(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, class_type = model)
        evaluation.eval()

        # Predictions
        predictions = Predictor(new_data=new_data, class_type=model, x_train=x_train, y_train = y_train)
        estimator = predictions.load_estimator(output_dir)
        predictions_applied = predictions.predict(estimator)
        predictions.export_pred_to_csv(predictions_applied)

    def save_best_parameters(self, best_parameters):
        with open(best_parameters_file_path, 'w') as best_parameters_file:
            json.dump(best_parameters, best_parameters_file)
            best_parameters_file.close()

    def load_best_parameters(self):
        with open(best_parameters_file_path) as best_parameters_file:
            return json.load(best_parameters_file)

    def read_dataframe(self, filename):
        return pd.read_csv(workspace + filename, encoding="ISO-8859-1")


if __name__ == '__main__':
    MainClass().main()
