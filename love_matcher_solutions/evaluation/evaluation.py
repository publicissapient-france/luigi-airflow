import warnings
import json
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import classification_report
from config.conf import *
from sklearn.externals import joblib



class Evaluator:
    warnings.filterwarnings("ignore")

    def __init__(self, x_train, y_train, x_test, y_test, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type

    def eval(self):
        print ("Evaluation...")
        reg = self.load_estimator(model_target=output_dir)
        y_true, y_pred = self.y_test, reg.predict(self.x_test)
        # TODO: 4.1 Apply classification report function on real and predicted values and write output to file
        # Solution
        evaluation_report_test = classification_report(y_true, y_pred)
        with open(output_dir + "/" + self.model_type + "_eval.txt", "w") as text_file:
            text_file.write(evaluation_report_test)

        pass

    def load_estimator(self, model_target):
        loaded_estimator = joblib.load(model_target + '/' + self.model_type + '_model.pkl')
        return loaded_estimator
