import warnings
import json
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.externals import joblib



class Evaluator:
    warnings.filterwarnings("ignore")

    def __init__(self, x_train, y_train, x_test, y_test, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type

    def eval(self, output_dir):
        print ("Evaluation...")
        estimator = self.load_estimator(model_target=output_dir)
        y_true, y_pred = self.y_test, estimator.predict(self.x_test)
        # TODO: 4.1 Apply classification_report() function on true and predicted values

        with open(output_dir + self.model_type + "_eval.txt", "w") as text_file:
            # TODO: 4.2 Write report to file
            # TODO: Verify your solution by running test_evaluation.py
            text_file.write()
        pass

    def load_estimator(self, model_target):
        return joblib.load(model_target + '/' + self.model_type + '_model.pkl')
