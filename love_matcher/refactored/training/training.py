from sklearn import ensemble
from sklearn import tree
from sklearn.externals import joblib


class Trainer:
    def __init__(self, x_train, y_train, x_test, y_test, best_params, model_type):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.estimator = None
        self.model = None
        self.best_params = best_params
        self.model_type = model_type

    def build_best_estimator(self):
        if self.model_type == "simple":
            model = tree.DecisionTreeClassifier()
            self.estimator = model.fit(self.x_train, self.y_train)
        else:
            model = ensemble.RandomForestClassifier(**self.best_params)
            self.estimator = model.fit(self.x_train, self.y_train)
        return self.model, self.estimator

    def save_estimator(self, model_target):
        print ("Saving...")
        joblib.dump(self.estimator, model_target + '/my_model.pkl')

    def score_estimator_train(self):
        return self.estimator.score(self.x_train, self.y_train)

    def score_estimator_test(self):
        return self.estimator.score(self.x_test, self.y_test)

    def combiner_pipeline(self):
        self.estimator = self.build_best_estimator()[1]
        score_train = self.score_estimator_train()
        score_test = self.score_estimator_test()
        return self.estimator, score_train, score_test
