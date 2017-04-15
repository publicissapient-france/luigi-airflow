from sklearn import ensemble
from sklearn import tree
from sklearn.externals import joblib
import pickle

class Trainer:
    def __init__(self, x_train, y_train, x_test, y_test, best_params, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.estimator = None
        self.best_params = best_params
        self.model_type = model_type

    def build_best_estimator(self):
        if self.model_type == "Decision_Tree":
            model = tree.DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 30)
            self.estimator = model.fit(self.x_train, self.y_train)
            tree.export_graphviz(self.estimator, out_file='tree.dot')
        else:
            model = ensemble.RandomForestClassifier(**self.best_params)
            self.estimator = model.fit(self.x_train, self.y_train)
        return model, self.estimator

    def save_estimator(self, model_target):
        print ("Saving model...")
        model, estimator = self.build_best_estimator()
        joblib.dump(model, model_target + '/' + self.model_type + '_model.pkl')

    def score_estimator_train(self):
        return self.estimator.score(self.x_train, self.y_train)

    def score_estimator_test(self):
        return self.estimator.score(self.x_test, self.y_test)

    def combiner_pipeline(self):
        self.estimator = self.build_best_estimator()[1]
        score_train = self.score_estimator_train()
        score_test = self.score_estimator_test()
        return self.estimator, score_train, score_test
