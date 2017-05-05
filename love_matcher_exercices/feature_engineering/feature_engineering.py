class FeatureEngineering:
    """
    This class aims to load and clean the dataset.
    """

    def __init__(self, features, suffix_1="_me", suffix_2="_partner", label="match_perc"):
        self.features = features
        self.suffix_1 = suffix_1
        self.suffix_2 = suffix_2
        self.label = label

    def get_partner_features(self, df, train_set=True):
        df_partner = df.copy()
        if train_set:
            df_partner = df_partner.drop(['pid','match'], 1).drop_duplicates()
        else:
            df_partner = df_partner.drop(['pid'], 1).drop_duplicates()
        merged_datasets = df.merge(df_partner, how="inner", left_on="pid", right_on="iid",
                                   suffixes=(self.suffix_1, self.suffix_2))
        print (merged_datasets.columns.values)
        print (merged_datasets.shape)

        return merged_datasets, merged_datasets[self.process_features_names("_me", "_partner")]

    def process_features_names(self, suffix_1, suffix_2):
        features_me = [feature + suffix_1 for feature in self.features]
        features_partner = [feature + suffix_2 for feature in self.features]
        features_all = features_me + features_partner
        return features_all