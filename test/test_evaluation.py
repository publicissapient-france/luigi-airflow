import pandas as pd
import unittest
import os
import glob
from pandas.core.frame import DataFrame
from pandas.util.testing import assert_frame_equal

from config.conf import output_dir
from love_matcher_exercises.preprocessing.raw_data_preprocessing import RawSetProcessing
from love_matcher_exercises.run.main import MainClass
from love_matcher_exercises.training.training import Trainer


class EvaluationTest(unittest.TestCase):
    def test_evaluation_result_has_been_successfully_saved(self):
        model_type = "Decision_Tree"
        output_dir_content = glob.glob(output_dir + "*")
        for f in output_dir_content:
            os.remove(f)
        MainClass(model_type=model_type).main()
        self.assertTrue(os.path.exists(output_dir + model_type + '_eval.txt'))


if __name__ == '__main__':
    unittest.main()
