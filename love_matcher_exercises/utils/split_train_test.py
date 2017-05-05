from sklearn.model_selection import train_test_split


class SplitTestTrain:

    def __init__(self, feat_eng_df, processed_features_names):
        self.feat_eng_df = feat_eng_df
        self.explanatory = processed_features_names

    def create_train_test_splits(self):
        explanatory, explained = self.create_df_explained_explanatory("match")
        # TODO: 2.1 Split dataset into train (70% of observations) and test set (30% of observations) using train_test_split function



        # To verify your results print shapes of your dataset, it should be equal to (2452, 38) (5722, 38) (2452,) (5722,):
        # print (x_train.shape,x_test.shape,y_train.shape,y_test.shape)
        pass

    def create_df_explained_explanatory(self, label):
        explanatory = self.explanatory
        explained = self.feat_eng_df[label]
        return explanatory, explained
