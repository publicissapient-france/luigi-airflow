from pandas.core.frame import DataFrame


class RawSetProcessing:
    """
    This class aims to load and clean the dataset.
    """

    def __init__(self, features):
        self.features = features

    # Select variables to process and include in the model
    @staticmethod
    def subset_features(features, df):
        sel_vars_df = df[features]
        return sel_vars_df

    @staticmethod
    # Remove ids with missing values
    def remove_missing_values(df):
        # TODO : 1.1 Write function to remove missing values. Return dataset.
        pass


    @staticmethod
    def drop_duplicated_values(df):
        # TODO : 1.2 Write function to remove duplicated values. Return dataset.
        # TODO: Verify your solution by running test_preprocessing.py
        pass

    # Combine processing stages
    def combiner_pipeline(self, dataframe):
        print ("Preprocessing...")
        raw_dataset = dataframe
        subset_df = self.subset_features(self.features,raw_dataset)
        subset_no_dup_df = self.drop_duplicated_values(subset_df)
        subset_filled_df = self.remove_missing_values(subset_no_dup_df)
        return subset_filled_df
