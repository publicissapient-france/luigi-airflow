import unittest

import pandas as pd

from love_matcher_exercises.utils.split_train_test import SplitTestTrain


class TrainTestSplitTest(unittest.TestCase):
    def test_train_test_split(self):
        """

        :return:
        """
        # Given
        raw_data_input = {
            'iid': ['1', '2', '3', '4', '5',
                    '6', '7', '8', '9', '10'],
            'first_name': ['Sue', 'Maria', 'Sandra', 'Bill', 'Tom',
                           'Bruce', 'Sandra', 'Bill', 'Tom', 'Bruce'],
            'sport': ['foot', 'volley', 'volley', 'basket', 'swim',
                      'tv', 'volley', 'basket', 'swim', 'tv'],
            'match': ['4', '5', '6', '1', '2',
                      '3', '6', '1', '2', '3'], }

        raw_data_input_df = pd.DataFrame(raw_data_input, columns=['iid', 'first_name', 'sport', 'match'])

        # When
        split_train_test = SplitTestTrain(feat_eng_df=raw_data_input_df,
                                          processed_features_names=raw_data_input_df[['iid', 'first_name', 'sport']])
        x_train, x_test, y_train, y_test = split_train_test.create_train_test_splits()

        # Then
        self.assertEquals(len(x_train), 7)
        self.assertEquals(len(x_test), 3)
        self.assertEquals(len(y_train), 7)
        self.assertEquals(len(y_test), 3)


if __name__ == '__main__':
    unittest.main()
