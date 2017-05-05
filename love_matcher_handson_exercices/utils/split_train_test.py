from sklearn.model_selection import train_test_split


class SplitTestTrain:

    def __init__(self, feat_eng_df, processed_features_names):
        self.feat_eng_df = feat_eng_df
        self.explanatory = processed_features_names

    def create_train_test_splits(self):
        explanatory, explained = self.create_df_explained_explanatory("match")
        x_train, x_test, y_train, y_test = train_test_split(explanatory, explained,
                                                            test_size=0.5, random_state=0,
                                                            stratify=explained)
        return x_train, x_test, y_train, y_test

    def create_df_explained_explanatory(self, label):
        # Tuning
        explanatory = self.explanatory
        if 'Unnamed: 0' in explanatory.columns:
            del explanatory['Unnamed: 0']
        explained = self.feat_eng_df[label]
        return explanatory, explained
