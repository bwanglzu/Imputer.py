"""Test imputer."""
import os
import unittest
import numpy as np
import pandas as pd
from imputer.imputer import Imputer


class TestImputer(unittest.TestCase):
    """imputer test class."""

    def test_imputer(self):
        """Test imputer function."""
        path = os.path.dirname(__file__) + "/train"
        impute = Imputer()
        cols = ['age', 'work_class', 'education',
                'num_edu_years', 'marital_status', 'occupation',
                'relationship', 'race', 'sex',
                'income', 'loss', 'hour',
                'country', 'label']
        X = pd.read_csv(path, names=cols, header=None)
        impute_first_column = impute.knn(X, 0)
        impute_eighth_column = impute.knn(X, 8, k=10, is_categorical=True)
        self.assertIsNotNone(impute_first_column)
        self.assertIsNotNone(impute_eighth_column)
        self.assertIsInstance(impute_first_column, np.ndarray)
        self.assertIsInstance(impute_eighth_column, np.ndarray)
        # column has no missing values
        impute_first_column = impute.knn(impute_first_column, 0)
        self.assertIsNotNone(impute_first_column)
        self.assertIsInstance(impute_first_column, np.ndarray)
        # user specify a column name
        impute_second_column = impute.knn(X, 'work_class')
        self.assertIsNotNone(impute_second_column)
        self.assertIsInstance(impute_second_column, np.ndarray)
        # user input is a ndarray
        X = np.genfromtxt(path, delimiter=',')
        impute_third_column = impute.knn(X, 2)
        self.assertIsNotNone(impute_third_column)
        self.assertIsInstance(impute_third_column, np.ndarray)


if __name__ == '__main__':
    unittest.main()
