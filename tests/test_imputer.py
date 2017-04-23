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
        X = pd.read_csv(path)
        impute_first_column = impute.knn(X, 0)
        impute_eighth_column = impute.knn(X, 8, k=10, is_categorical=True)
        self.assertIsNotNone(impute_first_column)
        self.assertIsNotNone(impute_eighth_column)
        self.assertIsInstance(impute_first_column, np.ndarray)
        self.assertIsInstance(impute_eighth_column, np.ndarray)


if __name__ == '__main__':
    unittest.main()
