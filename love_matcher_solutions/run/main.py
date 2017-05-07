import pandas as pd

from config.conf import *
from love_matcher_solutions.evaluation.evaluation import Evaluator
from love_matcher_solutions.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher_solutions.predictions.predictions import Predictor
from love_matcher_solutions.preprocessing.raw_data_preprocessing import RawSetProcessing
from love_matcher_solutions.training.training import Trainer
from love_matcher_solutions.utils.split_train_test import SplitTestTrain


class MainClass:
    def __init__(self, model_type):
        self.model_type = model_type

    def main(self):
        dataset = self.read_dataframe("Speed_Dating_Data.csv")
        new_data = self.read_dataframe("Submission_set.csv", sep=";")

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=dataset)

        # Feature engineering
        feature_engineering = FeatureEngineering(features=features)
        all_features_engineered_df, selected_features_df = feature_engineering.add_partner_features_train(dataset_df)
        print(all_features_engineered_df.shape)

        # Train
        split_test_train = SplitTestTrain(feat_eng_df=all_features_engineered_df,
                                          processed_features_names=selected_features_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()
        train = Trainer(x_train, y_train, x_test, y_test, model_type=self.model_type)
        train.save_estimator(output_dir)

        # Evaluation
        evaluation = Evaluator(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                               model_type=self.model_type)
        evaluation.eval(output_dir)

        # Predictions
        predictions = Predictor(new_data=new_data, model_type=self.model_type)
        estimator = predictions.load_estimator(output_dir)
        # Save predictions
        predictions.predict(estimator)

    def read_dataframe(self, filename, sep=","):
        return pd.read_csv(data_source + filename, sep=sep, encoding="ISO-8859-1")


if __name__ == '__main__':
    MainClass(model_type="Decision_Tree").main()
