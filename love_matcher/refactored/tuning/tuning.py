import warnings

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble


from love_matcher.refactored.split_test_train import SplitTestTrain

warnings.filterwarnings("ignore")


class TuneParameters:
    def __init__(self, feat_eng_df, parameters, scores, features):
        self.features = features
        self.feat_eng_df = feat_eng_df
        self.parameters = parameters
        self.scores = scores

    def tuning_parameters(self, trainset_x, trainset_y):
        rf_model = ensemble.RandomForestClassifier(n_estimators=5, class_weight="balanced", oob_score=False)

        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            grid_rfc = GridSearchCV(rf_model, self.parameters, n_jobs=10, cv=100, refit=True,
                                    scoring='%s_macro' % score)
            grid_rfc.fit(trainset_x, trainset_y)

            print("Best parameters set found on development set:")
            print("")
            print(grid_rfc.best_params_)
            print("")
            return grid_rfc, grid_rfc.best_estimator_.get_params()

    def combiner_pipeline(self):
        split_test_train = SplitTestTrain(self.feat_eng_df)
        x_train, x_test, y_train, y_test = split_test_train.create_train_test_splits()
        grid_rfc, best_params = self.tuning_parameters(x_train, y_train)
        return grid_rfc, best_params
