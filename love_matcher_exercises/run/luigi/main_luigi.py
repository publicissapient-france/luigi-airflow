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
        pass

    def requires(self):
        pass

    def run(self):
        pass


class EvaluationTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        pass

    def requires(self):
        pass

    def run(self):
        pass


class PredictionsTask(luigi.Task):
    model_type = luigi.Parameter()

    def output(self):
        pass

    def requires(self):
        pass

    def run(self):
        pass


class GeneratePredictions(luigi.WrapperTask):
    model_type = luigi.Parameter(default="Decision_Tree")

    def requires(self):
        pass


# TODO : luigi --module love_matcher_exercises.run.luigi.main_luigi GeneratePredictions