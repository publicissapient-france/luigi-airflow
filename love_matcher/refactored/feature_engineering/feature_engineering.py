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
            df_partner = df_partner.drop(['pid', 'match'], 1).drop_duplicates()
        else:
            df_partner = df_partner.copy()
        merged_datasets = df.merge(df_partner, how="inner", left_on="pid", right_on="iid",
                                   suffixes=(self.suffix_1, self.suffix_2))
        return merged_datasets, merged_datasets[self.process_features_names("_me", "_partner")]

    def combiner_pipeline(self, df):
        add_match_feat_df = self.add_success_failure_match(df)
        labels_df = self.label_to_categories(add_match_feat_df)
        model_set = self.aggregate_data(labels_df)
        return model_set

    def add_success_failure_match(self, df):
        df['total_match'] = df['match'].groupby(df['iid']).transform('sum')
        df['total_dates'] = df['match'].groupby(df['iid']).transform('count')
        df['total_nomatch'] = df['total_dates'] - df['total_match']
        df['match_perc'] = df['total_match'] / df['total_dates']
        return df

    def label_to_categories(self, df):
        df['match_success'] = pd.cut(df[self.label], bins=(0, 0.2, 1), include_lowest=True)
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
