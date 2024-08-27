import unittest
import os


class TestCase(unittest.TestCase):

    def test_add_one(self):
        from blixt_utils.utils import add_one
        strings = ['A', 'A1', 'A 1', 'A2', 'A 3']
        expected_results = ['A 1', 'A2', 'A 2', 'A3', 'A 4']
        for i, string in enumerate(strings):
            out = add_one(string)
            print(out)
            self.assertTrue(out == expected_results[i])

