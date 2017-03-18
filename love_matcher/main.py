import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

pd.set_option('display.max_columns', None)

local_path = "/home/dolounet/dev/workshops/"

local_filename = "Speed_Dating_Data.csv"

raw_dataset = pd.read_csv(local_path + local_filename, encoding="ISO-8859-1")


class RawSetProcessing(object):
    """
    This class aims to load and clean the dataset.
    """

    def __init__(self, source_path, filename, features):
        self.source_path = source_path
        self.filename = filename
        self.features = features

    # Load data
    def load_data(self):
        raw_dataset_df = pd.read_csv(self.source_path + self.filename, encoding="ISO-8859-1")
        return raw_dataset_df

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
        raw_dataset = self.load_data()
        subset_df = self.subset_features(raw_dataset)
        subset_no_dup_df = self.drop_duplicated_values(subset_df)
        subset_filled_df = self.remove_ids_with_missing_values(subset_no_dup_df)
        return subset_filled_df


my_variables_selection = ["iid", "pid", "match", "gender", "date", "go_out", "sports", "tvsports", "exercise", "dining",
                          "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies",
                          "concerts", "music", "shopping", "yoga"]

features = list(["gender", "date", "go_out", "sports", "tvsports", "exercise", "dining", "museums", "art",
                 "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music",
                 "shopping", "yoga"])

raw_set = RawSetProcessing(local_path, local_filename, my_variables_selection)
dataset_df = raw_set.combiner_pipeline()

suffix_me = "_me"
suffix_partner = "_partner"


def get_partner_features(df, suffix_1, suffix_2, ignore_vars=True):
    # print df[df["iid"] == 1]
    df_partner = df.copy()
    if ignore_vars is True:
        df_partner = df_partner.drop(['pid', 'match'], 1).drop_duplicates()
    else:
        df_partner = df_partner.copy()
    # print df_partner.shape
    merged_datasets = df.merge(df_partner, how="inner", left_on="pid", right_on="iid", suffixes=(suffix_1, suffix_2))
    # print merged_datasets[merged_datasets["iid_me"] == 1]
    return merged_datasets


feat_eng_df = get_partner_features(dataset_df, suffix_me, suffix_partner)


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


features_model = process_features_names(features, suffix_me, suffix_partner)

explanatory = feat_eng_df[features_model]
label = "match"

explained = feat_eng_df[label]
clf = tree.DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=10, max_depth=4)
clf = clf.fit(explanatory, explained)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(explanatory, explained, test_size=0.3, random_state=0)
parameters = [
    {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 10, 12, 14],
     'min_samples_split': [10, 20, 30], 'min_samples_leaf': [10, 15, 20]
     }
]

scores = ['precision', 'recall']
dtc = tree.DecisionTreeClassifier()

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print("")

    clf = GridSearchCV(dtc, parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print("")
    print(clf.best_params_)
    print("")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("")

best_param_dtc = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=10, min_samples_leaf=10,
                                             max_depth=14)
best_param_dtc = best_param_dtc.fit(explanatory, explained)

raw_dataset.rename(columns={"age_o": "age_of_partner", "race_o": "race_of_partner"}, inplace=True)
