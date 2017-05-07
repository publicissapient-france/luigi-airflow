from sklearn.model_selection import train_test_split


class SplitTestTrain:

    def __init__(self, feat_eng_df, processed_features_names):
        self.feat_eng_df = feat_eng_df
        self.explanatory = processed_features_names

    def create_train_test_splits(self):
        # Explanatory variables are used to model the explained variable
        explanatory, explained = self.create_df_explained_explanatory("match")
        # TODO: 2.1 Split dataset into train (70% of observations) and test set (30% of observations)
        #  using train_test_split function
        # TODO: Verify your solution by running test_split_train_test.py
        x_train, x_test, y_train, y_test = train_test_split(explanatory, explained, test_size=0.3)

        return x_train, x_test, y_train, y_test

    def create_df_explained_explanatory(self, label):
        explanatory = self.explanatory
        explained = self.feat_eng_df[label]
        return explanatory, explained
