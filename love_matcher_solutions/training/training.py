import os
from sklearn import tree
from sklearn.externals import joblib

from config.conf import *


class Trainer:
    def __init__(self, x_train, y_train, x_test, y_test, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type
        self.model = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf=30)

    def save_estimator(self, model_dir_path):
        print("Saving model...")
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)
        self.build_best_estimator()
        joblib.dump(self.model, model_dir_path + self.model_type + '_model.pkl')

    def build_best_estimator(self):
        print("Training...")
        # TODO: 3.1: Apply fit method of model object on train set,
        # TODO: verify you answer by running test_training.py
        self.model.fit(self.x_train, self.y_train)

        # Export model visualisation
        tree.export_graphviz(self.model, out_file=output_dir + "/tree.dot")
