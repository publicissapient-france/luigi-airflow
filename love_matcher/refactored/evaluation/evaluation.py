import warnings
import json
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import classification_report
from docs.conf import *

class Evaluator:
    warnings.filterwarnings("ignore")

    def __init__(self, x_train, y_train, x_test, y_test, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_type = model_type

    def get_best_estimator(self,json_path):
        with open(json_path) as data_file:
            best_estimator = json.load(data_file)
        return best_estimator

    def build_best_estimator(self,json_path):
        if self.model_type == 'Decision_Tree':
            reg = tree.DecisionTreeClassifier(random_state=1234)
        elif self.model_type == 'Random_Forest':
            best_estimator = self.get_best_estimator(json_path)
            params = best_estimator
            reg = ensemble.RandomForestClassifier(**params)
        else:
            return 0
        return reg.fit(self.x_train,self.y_train)

    def eval(self):
        reg = self.build_best_estimator(output_dir + "/" + self.model_type + "_best_parameters.json")
        print (output_dir + "/" + self.model_type + "_best_parameters.json")
        y_true, y_pred = self.y_test, reg.predict(self.x_test)
        evaluation_report_test = classification_report(y_true, y_pred)
        print (evaluation_report_test)
        with open(output_dir + "/" + self.model_type + "_eval.txt", "w") as text_file:
            text_file.write(evaluation_report_test)
