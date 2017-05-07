class FeatureEngineering:
    """
    Add new features to existing dataset
    """

    def __init__(self, features, suffix_1="_me", suffix_2="_partner", label="match_perc"):
        self.features = features
        self.suffix_1 = suffix_1
        self.suffix_2 = suffix_2
        self.label = label

    def add_partner_features_train(self, df):
        return self.add_partner_features(df, ['pid', 'match'])

    def add_partner_features_test(self, df):
        return self.add_partner_features(df, ['pid'])

    def add_partner_features(self, df, variables_to_drop):
        print("Feature engineering...")
        df_partner = df.copy()
        # Match variable exists only in trainset
        df_partner = df_partner.drop(variables_to_drop, 1).drop_duplicates()
        # Add partner features
        merged_datasets = df.merge(df_partner, how="inner", left_on="pid", right_on="iid",
                                   suffixes=(self.suffix_1, self.suffix_2))
        return merged_datasets, merged_datasets[self.process_features_names("_me", "_partner")]

    def process_features_names(self, suffix_1, suffix_2):
        features_me = [feature + suffix_1 for feature in self.features]
        features_partner = [feature + suffix_2 for feature in self.features]
        features_all = features_me + features_partner
        return features_all
