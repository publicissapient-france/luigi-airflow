import datetime
import json

import luigi
import pandas as pd
from config.conf import *
from love_matcher_exercises.evaluation.evaluation import Evaluator
from love_matcher_exercises.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher_exercises.predictions.predictions import Predictor
from love_matcher_exercises.preprocessing.raw_data_preprocessing import RawSetProcessing
from love_matcher_exercises.utils.split_train_test import SplitTestTrain
from love_matcher_exercises.training.training import Trainer


class FeatureEngineeringTask(luigi.Task):
    def output(self):
        pass
    def run(self):
        pass

    def read_dataframe(self):
        pass


class TrainTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(output_dir + '/' + str(self.model_type) + '_model.pkl')

    def requires(self):
        # TODO 6.1
        return FeatureEngineeringTask(self.model_type)

    def run(self):
        feat_eng_df = pd.read_csv(feature_engineered_dataset_file_path)
        processed_features_names_df = pd.read_csv(processed_features_names_file_path)

        split_test_train = SplitTestTrain(feat_eng_df=feat_eng_df, processed_features_names=processed_features_names_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()

        train = Trainer(x_train, y_train, x_test, y_test, model_type=str(self.model_type))
        train.save_estimator(output_dir)
        estimator = train.build_best_estimator()
        print(estimator)


class EvaluationTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        # TODO 6.2
        return luigi.LocalTarget(output_dir + '/' + str(self.model_type) + '_eval.txt')

    def requires(self):
        return TrainTask(self.model_type)

    def run(self):
        feat_eng_df = pd.read_csv(feature_engineered_dataset_file_path)
        processed_features_names_df = pd.read_csv(processed_features_names_file_path)

        split_test_train = SplitTestTrain(feat_eng_df=feat_eng_df, processed_features_names=processed_features_names_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()

        evaluation = Evaluator(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                               model_type=str(self.model_type))
        evaluation.eval()


class PredictionsTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(output_dir + "/" + str(self.model_type) + "_predictions.csv")

    def requires(self):
        # TODO 6.3
        return EvaluationTask(self.model_type)

    def run(self):
        new_data = pd.read_csv(data_source + "Submission_set.csv", encoding="ISO-8859-1", sep=";")
        predictions = Predictor(new_data=new_data, model_type=str(self.model_type))
        estimator = predictions.load_estimator(output_dir)
        predictions.predict(estimator)


class GeneratePredictions(luigi.WrapperTask):
    model_type = luigi.Parameter(default="Decision_Tree")

    def requires(self):
        return FeatureEngineeringTask(), \
               TrainTask(model_type=self.model_type), \
               EvaluationTask(model_type=self.model_type), \
               PredictionsTask(model_type=self.model_type)


# TODO : luigi --module love_matcher_exercises.run.luigi.main_luigi GeneratePredictions