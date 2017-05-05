import pandas as pd


class FeatureEngineering:
    """
    This class aims to load and clean the dataset.
    """

    def __init__(self, features, suffix_1="_me", suffix_2="_partner", label="match_perc"):
        self.features = features
        self.suffix_1 = suffix_1
        self.suffix_2 = suffix_2
        self.label = label

    def get_partner_features(self, df, ignore_vars=True):
        df_partner = df.copy()
        if ignore_vars is True:
            if 'match' in df_partner.columns:
                del df_partner['match']
            df_partner = df_partner.drop(['pid'], 1).drop_duplicates()
        else:
            df_partner = df_partner.copy()
        merged_datasets = df.merge(df_partner, how="inner", left_on="pid", right_on="iid",
                                   suffixes=(self.suffix_1, self.suffix_2))
        return merged_datasets, merged_datasets[self.process_features_names("_me", "_partner")]

    def add_average_sport(self, df):
        # TODO : 2.1 Ajouter une nouvelle variable moyenne des notes pour les sports, verification: test unitaire
        #df['average_sport'] = df['match'].groupby(df['iid']).transform('mean')
        return df

    def process_features_names(self, suffix_1, suffix_2):
        features_me = [feature + suffix_1 for feature in self.features]
        features_partner = [feature + suffix_2 for feature in self.features]
        features_all = features_me + features_partner
        return features_all

    @staticmethod
    def aggregate_data(df):
        model_set = df.drop(["pid", "match"], 1)
        model_set = model_set.drop_duplicates()
        return model_set
