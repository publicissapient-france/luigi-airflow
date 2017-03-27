from sklearn import ensemble


class Trainer:
    def __init__(self, x_train, y_train, x_test, y_test, best_params):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.estimator = None
        self.best_params = best_params

    def build_best_estimator(self):
        model = ensemble.RandomForestClassifier(**self.best_params)
        self.estimator = model.fit(self.x_train, self.y_train)
        return self.estimator

    def score_estimator_train(self):
        return self.estimator.score(self.x_train, self.y_train)

    def score_estimator_test(self):
        return self.estimator.score(self.x_test, self.y_test)

    def combiner_pipeline(self):
        self.estimator = self.build_best_estimator()
        score_train = self.score_estimator_train()
        score_test = self.score_estimator_test()
        return self.estimator, score_train, score_test