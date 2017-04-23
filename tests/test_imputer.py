"""Test imputer."""
import unittest
import sys
sys.path.append('..')
from imputer.imputer import Imputer


class TestImputer(unittest.TestCase):
    """imputer test class."""

    def test_imputer(self):
        """Test imputer function."""
        impute = Imputer()
        type(impute)
        pass


if __name__ == '__main__':
    unittest.main()
