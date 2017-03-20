import pandas as pd
from sklearn import ensemble
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


class MainClass:
    def __init__(self, workspace="/home/dolounet/dev/workshops/"):
        self.workspace = workspace

    def main(self):
        # TODO make workspace configurable from luigi.cfg
        raw_dataset = MainClass.read_dataframe(self=self.workspace)

        # TODO make variables externalized
        my_variables_selection = ["iid", "pid", "match", "gender", "date", "go_out", "sports", "tvsports", "exercise",
                                  "dining",
                                  "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater",
                                  "movies",
                                  "concerts", "music", "shopping", "yoga"]
        # TODO make features externalized
        features = list(["gender", "date", "go_out", "sports", "tvsports", "exercise", "dining", "museums", "art",
                         "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music",
                         "shopping", "yoga"])

        # Preprocessing
        raw_dataset = RawSetProcessing(my_variables_selection, dataframe=raw_dataset)
        dataset_df = raw_dataset.combiner_pipeline()

        suffix_me = "_me"
        suffix_partner = "_partner"

        my_label = "match_perc"
        label = "match"

        # Feature engineering
        feature_engineering = FeatureEngineering(suffix_1=suffix_me, suffix_2=suffix_partner, label=my_label)
        feat_eng_df = feature_engineering.get_partner_features(dataset_df)

        features_model = self.process_features_names(features, suffix_me, suffix_partner)



        # Parameters for Random Forest

        # TODO make random forest parameters externalized
        parameters = [
            {'max_depth': [8, 10, 12, 14, 16, 18],
             'min_samples_split': [10, 15, 20, 25, 30],
             'min_samples_leaf': [10, 15, 20, 25, 30]
             }
        ]
        scores = ['precision', 'recall']
        rf_model = ensemble.RandomForestClassifier(n_estimators=5, oob_score=False)

        # Tuning
        explanatory = feat_eng_df[features_model]
        explained = feat_eng_df[label]
        tune = TuneParameters(explanatory, explained, rf_model, parameters, scores)
        best_parameters = tune.combiner_pipeline()
        x_train, x_test, y_train, y_test = tune.create_train_test_splits()

        # Train
        train = Trainer(x_train, y_train, x_test, y_test, best_parameters)
        estimator, score_train, score_test = train.combiner_pipeline()


    def read_dataframe(self):
        return pd.read_csv(self.workspace + "Speed_Dating_Data.csv", encoding="ISO-8859-1")

    def process_features_names(self, features, suffix_1, suffix_2):
        features_me = [feat + suffix_1 for feat in features]
        features_partner = [feat + suffix_2 for feat in features]
        features_all = features_me + features_partner
        return features_all


class RawSetProcessing:
    """
    This class aims to load and clean the dataset.
    """

    def __init__(self, features, dataframe):
        self.features = features
        self.dataframe = dataframe

    # Select variables to process and include in the model
    def subset_features(self, df):
        sel_vars_df = df[self.features]
        return sel_vars_df

    @staticmethod
    # Remove ids with missing values
    def remove_ids_with_missing_values(df):
        sel_vars_filled_df = df.dropna()
        return sel_vars_filled_df

    @staticmethod
    def drop_duplicated_values(df):
        df = df.drop_duplicates()
        return df

    # Combine processing stages
    def combiner_pipeline(self):
        raw_dataset = self.dataframe
        subset_df = self.subset_features(raw_dataset)
        subset_no_dup_df = self.drop_duplicated_values(subset_df)
        subset_filled_df = self.remove_ids_with_missing_values(subset_no_dup_df)
        return subset_filled_df


# add suffix to each element of list


class FeatureEngineering:
    """
    This class aims to load and clean the dataset.
    """

    def __init__(self, suffix_1, suffix_2, label):
        self.suffix_1 = suffix_1
        self.suffix_2 = suffix_2
        self.label = label

    def combiner_pipeline(self, df):
        add_match_feat_df = self.add_success_failure_match(df)
        labels_df = self.label_to_categories(add_match_feat_df)
        model_set = self.aggregate_data(labels_df)
        return model_set

    def get_partner_features(self, df, ignore_vars=True):
        df_partner = df.copy()
        if ignore_vars is True:
            df_partner = df_partner.drop(['pid', 'match'], 1).drop_duplicates()
        else:
            df_partner = df_partner.copy()
        merged_datasets = df.merge(df_partner, how="inner", left_on="pid", right_on="iid",
                                   suffixes=(self.suffix_1, self.suffix_2))
        return merged_datasets

    def add_success_failure_match(self, df):
        df['total_match'] = df['match'].groupby(df['iid']).transform('sum')
        df['total_dates'] = df['match'].groupby(df['iid']).transform('count')
        df['total_nomatch'] = df['total_dates'] - df['total_match']
        df['match_perc'] = df['total_match'] / df['total_dates']
        return df

    def label_to_categories(self, df):
        df['match_success'] = pd.cut(df[self.label], bins=(0, 0.2, 1), include_lowest=True)
        return df

    @staticmethod
    def aggregate_data(df):
        model_set = df.drop(["pid", "match"], 1)
        model_set = model_set.drop_duplicates()
        return model_set


class TuneParameters:
    def __init__(self, explanatory_vars, explained_var, estimator, parameters, scores):
        self.explanatory_vars = explanatory_vars
        self.explained_var = explained_var
        self.estimator = estimator
        self.parameters = parameters
        self.scores = scores

    def create_train_test_splits(self):
        X_train, X_test, y_train, y_test = train_test_split(self.explanatory_vars, self.explained_var,
                                                            test_size=0.5, random_state=0,
                                                            stratify=self.explained_var)
        return X_train, X_test, y_train, y_test

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
        X_train, X_test, y_train, y_test = self.create_train_test_splits()
        best_params = self.tuning_parameters(X_train, X_test, y_train, y_test)
        return best_params


class Trainer:
    def __init__(self, x_train, y_train, x_test, y_test, best_params):
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.estimator = None
        self.best_params = best_params

    def build_best_estimator(self):
        params = self.best_params
        model = ensemble.RandomForestClassifier(**params)
        self.estimator = model.fit(self.X_train, self.y_train)
        return self.estimator

    def score_estimator_train(self):
        return self.estimator.score(self.X_train, self.y_train)

    def score_estimator_test(self):
        return self.estimator.score(self.X_test, self.y_test)

    def combiner_pipeline(self):
        self.estimator = self.build_best_estimator()
        score_train = self.score_estimator_train()
        score_test = self.score_estimator_test()
        return self.estimator, score_train, score_test
