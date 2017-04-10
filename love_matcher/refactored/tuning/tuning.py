import warnings

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from love_matcher.refactored.split_test_train import SplitTestTrain

warnings.filterwarnings("ignore")


class TuneParameters:
    def __init__(self, feat_eng_df, estimator, parameters, scores, features):
        self.features = features
        self.feat_eng_df = feat_eng_df
        self.estimator = estimator
        self.parameters = parameters
        self.scores = scores

    def tuning_parameters(self, trainset_x, testset_x, trainset_y, testset_y):
        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            grid_rfc = GridSearchCV(self.estimator, self.parameters, n_jobs=10, cv=100, refit=True,
                                    scoring='%s_macro' % score)
            grid_rfc.fit(trainset_x, trainset_y)

            print("Best parameters set found on development set:")
            print("")
            print(grid_rfc.best_params_)
            print("")
            # y_true, y_pred = testset_y, grid_rfc.predict(testset_x)
            # print(classification_report(y_true, y_pred))
            # print("")

            return grid_rfc, grid_rfc.best_estimator_.get_params()

    def combiner_pipeline(self):
        split_test_train = SplitTestTrain(self.feat_eng_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()
        grid_rfc, best_params = self.tuning_parameters(x_train, x_test, y_train, y_test)
        return grid_rfc, best_params
