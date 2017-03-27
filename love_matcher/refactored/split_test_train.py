from sklearn.model_selection import train_test_split


class SplitTestTrain:

    def __init__(self, feat_eng_df, features):
        self.features = features
        self.feat_eng_df = feat_eng_df

    def create_train_test_splits(self):
        explanatory, explained = self.create_df_explained_explanatory("match")
        x_train, x_test, y_train, y_test = train_test_split(explanatory, explained,
                                                            test_size=0.5, random_state=0,
                                                            stratify=explained)
        return x_train, x_test, y_train, y_test

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
