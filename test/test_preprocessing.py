import pandas as pd
import unittest
from pandas.util.testing import assert_frame_equal

from love_matcher_exercises.preprocessing.raw_data_preprocessing import RawSetProcessing


class RawSetProcessingTest(unittest.TestCase):

    def test_remove_missing_values(self):
        """

        :return:
        """
        # Given
        raw_data_input = {
            'iid': ['1', '2', '3', '4', '5', '6'],
            'first_name': ['Sue', 'Maria', 'Sandra', 'Bill', None, 'Bruce'],
            'sport': ['foot', None, 'volley', 'basket', 'swim', 'tv'],
            'pid': ['4', '5', '6', '1', '2', '3'], }

        df_a = pd.DataFrame(raw_data_input, columns=['iid', 'first_name', 'sport', 'pid'])

        raw_data_output = {
            'iid': ['1', '3', '4', '6'],
            'first_name': ['Sue', 'Sandra', 'Bill', 'Bruce'],
            'sport': ['foot', 'volley', 'basket', 'tv'],
            'pid': ['4', '6', '1', '3'], }

        expected_output_values = pd.DataFrame(raw_data_output, columns=['iid', 'first_name', 'sport', 'pid'])

        # When
        preprocessing = RawSetProcessing(features=None)
        output_values = preprocessing.remove_missing_values(df_a)

        # Then
        assert_frame_equal(output_values.reset_index(drop=True), expected_output_values.reset_index(drop=True))

    def test_drop_duplicated_values(self):
        """

        :return:
        """
        # Given
        raw_data_input = {
            'iid': ['1', '1', '3', '4', '6', '6'],
            'first_name': ['Sue', 'Sue', 'Sandra', 'Bill', 'Bruce', 'Bruce'],
            'sport': ['foot', 'foot', 'volley', 'basket', 'tv', 'tv'],
            'pid': ['4', '4', '6', '1', '3', '3'], }

        df_a = pd.DataFrame(raw_data_input, columns=['iid', 'first_name', 'sport', 'pid'])

        raw_data_output = {
            'iid': ['1', '3', '4', '6'],
            'first_name': ['Sue', 'Sandra', 'Bill', 'Bruce'],
            'sport': ['foot', 'volley', 'basket', 'tv'],
            'pid': ['4', '6', '1', '3'], }

        expected_output_values = pd.DataFrame(raw_data_output, columns=['iid', 'first_name', 'sport', 'pid'])

        # When
        preprocessing = RawSetProcessing(features=None)
        output_values = preprocessing.drop_duplicated_values(df_a)

        # Then
        assert_frame_equal(output_values.reset_index(drop=True), expected_output_values.reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
