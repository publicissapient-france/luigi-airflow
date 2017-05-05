from sklearn import tree
from sklearn.externals import joblib
from config.conf import *

class Trainer:
    def __init__(self, x_train, y_train, x_test, y_test, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.estimator = None
        self.model_type = model_type

    def build_best_estimator(self):
        model = tree.DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 30)
        self.estimator = model.fit(self.x_train, self.y_train)
        # TODO 2.1: appliquer la methode fit du modele sur le train set
        tree.export_graphviz(self.estimator, out_file= output_dir + "/tree.dot")
        return model, self.estimator

    def save_estimator(self, model_target):
        print ("Saving model...")
        model, estimator = self.build_best_estimator()
        joblib.dump(model, model_target + '/' + self.model_type + '_model.pkl')

    def combiner_pipeline(self):
        self.estimator = self.build_best_estimator()[1]
        return self.estimator
