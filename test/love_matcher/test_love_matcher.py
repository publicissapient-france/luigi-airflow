import pandas as pd
import unittest

from love_matcher.main import get_partner_features


class FeatureEngineeringTest(unittest.TestCase):
    def test_get_partner_features(self):
        """

        :return:
        """
        # Given
        raw_data_a = {
            'iid': ['1', '2', '3', '4', '5', '6'],
            'first_name': ['Sue', 'Maria', 'Sandra', 'Bill', 'Brian', 'Bruce'],
            'sport': ['foot', 'run', 'volley', 'basket', 'swim', 'tv'],
            'pid': ['4', '5', '6', '1', '2', '3'], }

        df_a = pd.DataFrame(raw_data_a, columns=['iid', 'first_name', 'sport', 'pid'])

        expected_output_values = pd.DataFrame({
            'iid_me': ['1', '2', '3', '4', '5', '6'],
            'first_name_me': ['Sue', 'Maria', 'Sandra', 'Bill', 'Brian', 'Bruce'],
            'sport_me': ['foot', 'run', 'volley', 'basket', 'swim', 'tv'],
            'pid_me': ['4', '5', '6', '1', '2', '3'],
            'iid_partner': ['4', '5', '6', '1', '2', '3'],
            'first_name_partner': ['Bill', 'Brian', 'Bruce', 'Sue', 'Maria', 'Sandra'],
            'sport_partner': ['basket', 'swim', 'tv', 'foot', 'run', 'volley'],
            'pid_partner': ['1', '2', '3', '4', '5', '6']
        }, columns=['iid_me', 'first_name_me', 'sport_me', 'pid_me',
                    'iid_partner', 'first_name_partner', 'sport_partner', 'pid_partner'])

        # When

        output_values = get_partner_features(df_a, "_me", "_partner", ignore_vars=False)
        print(output_values)

        # Then

        self.assertItemsEqual(output_values, expected_output_values)
