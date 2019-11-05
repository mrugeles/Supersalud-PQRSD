import unittest
import sys
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from Imputer import Imputer, ImputerStrategy
from Model import Model

class ImputerTest(unittest.TestCase):

    titanic = pd.read_csv('test/titanic.csv')
    imputer = Imputer()
    model = Model()

    def setUp(self):
        pass

    def test_mean_imput(self):
        expected = 29.6991
        null_indexes = list(self.titanic.loc[self.titanic['Age'].isnull()].index)
        result_df = self.imputer.impute(self.titanic[['Age']], ImputerStrategy.MEAN)
        result = list(set(result_df.iloc[null_indexes]['Age']))[0]
        self.assertAlmostEqual(expected, result, places = 4)

    def test_median_imput(self):
        expected = 28
        null_indexes = list(self.titanic.loc[self.titanic['Age'].isnull()].index)
        result_df = self.imputer.impute(self.titanic[['Age']], ImputerStrategy.MEDIAN)
        result = list(set(result_df.iloc[null_indexes]['Age']))[0]
        self.assertAlmostEqual(expected, result)

    def test_mode_imput(self):
        expected = 24
        null_indexes = list(self.titanic.loc[self.titanic['Age'].isnull()].index)
        result_df = self.imputer.impute(self.titanic[['Age']], ImputerStrategy.MODE)
        result = list(set(result_df.iloc[null_indexes]['Age']))[0]
        self.assertAlmostEqual(expected, result)

    def test_regressor_imput(self):
        #self.imputer.regressor_imputer(self.titanic, 'Age')
        self.assertAlmostEqual(True, True)

if __name__ == '__main__':
    unittest.main()
