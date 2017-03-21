from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


class TuneParameters:
    def __init__(self, explanatory_vars, explained_var, estimator, parameters, scores):
        self.explanatory_vars = explanatory_vars
        self.explained_var = explained_var
        self.estimator = estimator
        self.parameters = parameters
        self.scores = scores

    def create_train_test_splits(self):
        x_train, x_test, y_train, y_test = train_test_split(self.explanatory_vars, self.explained_var,
                                                            test_size=0.5, random_state=0,
                                                            stratify=self.explained_var)
        return x_train, x_test, y_train, y_test

    def tuning_parameters(self, trainset_x, testset_x, trainset_y, testset_y):
        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            grid_rfc = GridSearchCV(self.estimator, self.parameters, n_jobs=100, cv=10, refit=True,
                                    scoring='%s_macro' % score)
            grid_rfc.fit(trainset_x, trainset_y)

            print("Best parameters set found on development set:")
            print("")
            print(grid_rfc.best_params_)
            print("")
            y_true, y_pred = testset_y, grid_rfc.predict(testset_x)
            print(classification_report(y_true, y_pred))
            print("")

            best_parameters = grid_rfc.best_estimator_.get_params()
            return best_parameters

    def combiner_pipeline(self):
        x_train, x_test, y_train, y_test = self.create_train_test_splits()
        best_params = self.tuning_parameters(x_train, x_test, y_train, y_test)
        return best_params