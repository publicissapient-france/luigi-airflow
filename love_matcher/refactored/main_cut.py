import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


class MainClass:
    @staticmethod
    def main():
        raw_dataset = MainClass.read_dataframe(workspace="/home/dolounet/dev/workshops/")

        my_variables_selection = ["iid", "pid", "match", "gender", "date", "go_out", "sports", "tvsports", "exercise",
                                  "dining",
                                  "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater",
                                  "movies",
                                  "concerts", "music", "shopping", "yoga"]

        features = list(["gender", "date", "go_out", "sports", "tvsports", "exercise", "dining", "museums", "art",
                         "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music",
                         "shopping", "yoga"])

        raw_dataset = RawSetProcessing(my_variables_selection, dataframe=raw_dataset)
        dataset_df = raw_dataset.combiner_pipeline()

        suffix_me = "_me"
        suffix_partner = "_partner"

        features_model = process_features_names(features, suffix_me, suffix_partner)

        feat_eng_df = get_partner_features(dataset_df, suffix_me, suffix_partner)
        explanatory = feat_eng_df[features_model]
        label = "match"

        explained = feat_eng_df[label]
        clf = tree.DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=10, max_depth=4)
        clf = clf.fit(explanatory, explained)

        # Split the dataset in two equal parts
        x_train, x_test, y_train, y_test = train_test_split(explanatory, explained, test_size=0.3, random_state=0)
        parameters = [
            {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 10, 12, 14],
             'min_samples_split': [10, 20, 30], 'min_samples_leaf': [10, 15, 20]
             }
        ]

        scores = ['precision', 'recall']
        dtc = tree.DecisionTreeClassifier()

        best_param_dtc = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=10, min_samples_leaf=10,
                                                     max_depth=14)
        best_param_dtc = best_param_dtc.fit(explanatory, explained)

        raw_dataset.rename(columns={"age_o": "age_of_partner", "race_o": "race_of_partner"}, inplace=True)
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print("")

            clf = GridSearchCV(dtc, parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(x_train, y_train)

            print("Best parameters set found on development set:")
            print("")
            print(clf.best_params_)
            print("")
            y_true, y_pred = y_test, clf.predict(x_test)
            print(classification_report(y_true, y_pred))
            print("")

    @staticmethod
    def read_dataframe(workspace):
        return pd.read_csv(workspace + "Speed_Dating_Data.csv", encoding="ISO-8859-1")


class RawSetProcessing(object):
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


def get_partner_features(df, suffix_1, suffix_2, ignore_vars=True):
    df_partner = df.copy()
    if ignore_vars is True:
        df_partner = df_partner.drop(['pid', 'match'], 1).drop_duplicates()
    else:
        df_partner = df_partner.copy()
    merged_datasets = df.merge(df_partner, how="inner", left_on="pid", right_on="iid", suffixes=(suffix_1, suffix_2))
    return merged_datasets


# add suffix to each element of list
def process_features_names(features, suffix_1, suffix_2):
    print(features)
    print(suffix_1)
    print(suffix_2)
    features_me = [feat + suffix_1 for feat in features]
    features_partner = [feat + suffix_2 for feat in features]
    features_all = features_me + features_partner
    print(features_all)
    return features_all
