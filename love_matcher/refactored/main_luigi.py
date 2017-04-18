import datetime
import json

import luigi
import pandas as pd
from docs.conf import *
from love_matcher.refactored.evaluation.evaluation import Evaluator
from love_matcher.refactored.feature_engineering.feature_engineering import FeatureEngineering
from love_matcher.refactored.predictions.predictions import Predictor
from love_matcher.refactored.preprocessing.raw_set_processing import RawSetProcessing
from love_matcher.refactored.split_test_train import SplitTestTrain
from love_matcher.refactored.training.training import Trainer
from love_matcher.refactored.tuning.tuning import TuneParameters


class FeatureEngineeringTask(luigi.Task):
    def output(self):
        return luigi.LocalTarget(feature_engineered_dataset_file_path), \
               luigi.LocalTarget(processed_features_names_file_path)

    def run(self):
        dataset = self.read_dataframe()

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection)
        dataset_df = raw_dataset.combiner_pipeline(dataframe=dataset)

        # Feature engineering
        feature_engineering = FeatureEngineering(features=features)
        feat_eng_df, processed_features_names = feature_engineering.get_partner_features(dataset_df)
        feat_eng_df.to_csv(feature_engineered_dataset_file_path)
        processed_features_names.to_csv(processed_features_names_file_path)

    def read_dataframe(self):
        return pd.read_csv(workspace + "Speed_Dating_Data.csv", encoding="ISO-8859-1")


class TuneTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(output_dir + "/" + str(self.model_type) + "_best_parameters.json")

    def requires(self):
        return FeatureEngineeringTask()

    def run(self):
        # Tuning
        feat_eng_df = pd.read_csv(feature_engineered_dataset_file_path)
        processed_features_names_df = pd.read_csv(processed_features_names_file_path)
        if self.model_type == "Decision_Tree":
            return 0
        else:
            tune = TuneParameters(feat_eng_df, processed_features_names_df, parameters, scores, features)
            best_parameters = tune.combiner_pipeline()[1]
        self.save_best_parameters(best_parameters)

    def save_best_parameters(self, best_parameters):
        with open(output_dir + "/" + str(self.model_type) + "_best_parameters.json", 'w') as best_parameters_file:
            json.dump(best_parameters, best_parameters_file)
            best_parameters_file.close()


class TrainTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(output_dir + '/' + str(self.model_type) + '_model.pkl')

    def requires(self):
        if self.model_type == "Decision_Tree":
            self.start_time = datetime.datetime.now()
            return FeatureEngineeringTask()
        else:
            self.start_time = datetime.datetime.now()
            return TuneTask(self.model_type)

    def run(self):
        feat_eng_df = pd.read_csv(feature_engineered_dataset_file_path)
        processed_features_names_df = pd.read_csv(processed_features_names_file_path)
        if self.model_type == "Decision_Tree":
            best_parameters = None
        else:
            best_parameters = self.load_best_parameters()
        split_test_train = SplitTestTrain(feat_eng_df=feat_eng_df, processed_features_names=processed_features_names_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()
        train = Trainer(x_train, y_train, x_test, y_test, best_parameters, model_type=str(self.model_type))
        train.save_estimator(output_dir)
        estimator, score_train, score_test = train.combiner_pipeline()
        print(estimator, score_train, score_test)
        end_time = datetime.datetime.now()
        print("Total time spent %s" % (end_time - self.start_time))

    def load_best_parameters(self):
        with open(output_dir + "/" + str(self.model_type) + "_best_parameters.json") as best_parameters_file:
            return json.load(best_parameters_file)


class EvaluationTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(output_dir + "/" + str(self.model_type) + "_eval.txt")

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
        return TrainTask(self.model_type)

    def run(self):
        new_data = pd.read_csv(workspace + "Submission_set.csv", encoding="ISO-8859-1", sep=";")
        predictions = Predictor(new_data=new_data, model_type=str(self.model_type))
        estimator = predictions.load_estimator(output_dir)
        predictions_applied = predictions.predict(estimator)
        predictions.export_pred_to_csv(predictions_applied)


class GenerateFirstModel(luigi.WrapperTask):
    model_type = luigi.Parameter(default="Decision_Tree")

    def requires(self):
        return FeatureEngineeringTask(), \
               TrainTask(model_type=self.model_type), \
               EvaluationTask(model_type=self.model_type), \
               PredictionsTask(model_type=self.model_type)


class GenerateSecondModel(luigi.WrapperTask):
    model_type = luigi.Parameter(default="Random_Forest")

    def requires(self):
        return FeatureEngineeringTask(), \
               TuneTask(model_type=self.model_type), \
               TrainTask(model_type=self.model_type), \
               EvaluationTask(model_type=self.model_type), \
               PredictionsTask(model_type=self.model_type)
