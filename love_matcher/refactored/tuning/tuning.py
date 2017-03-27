from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

warnings.filterwarnings("ignore")

class TuneParameters:
    def __init__(self, feat_eng_df, estimator, parameters, scores, features):
        self.features = features
        self.feat_eng_df = feat_eng_df
        self.estimator = estimator
        self.parameters = parameters
        self.scores = scores

    def create_df_explained_explanatory(self, label):
        features_model = self.process_features_names(self.features, "_me", "_partner")

        # Tuning
        explanatory = self.feat_eng_df[features_model]
        explained = self.feat_eng_df[label]
        return explanatory, explained


    def process_features_names(self, features, suffix_1, suffix_2):
        features_me = [feat + suffix_1 for feat in features]
        features_partner = [feat + suffix_2 for feat in features]
        features_all = features_me + features_partner
        return features_all

    def create_train_test_splits(self):
        explanatory, explained = self.create_df_explained_explanatory("match")
        x_train, x_test, y_train, y_test = train_test_split(explanatory, explained,
                                                            test_size=0.5, random_state=0,
                                                            stratify=explained)
        return x_train, x_test, y_train, y_test

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
            y_true, y_pred = testset_y, grid_rfc.predict(testset_x)
            print(classification_report(y_true, y_pred))
            print("")

            return grid_rfc.best_estimator_.get_params()

    def combiner_pipeline(self):
        x_train, x_test, y_train, y_test = self.create_train_test_splits()
        best_params = self.tuning_parameters(x_train, x_test, y_train, y_test)
        return best_params