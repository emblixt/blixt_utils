import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from blixt_utils.io.io import read_general_ascii_GENERAL as read_file

test_file_dir = os.path.dirname(__file__).replace(
    'blixt_utils\\blixt_utils\\unit_tests',
    'blixt_rp\\test_data')

las_file1 = os.path.join(test_file_dir, "L-30.las")
petrel_well_tops_file = "C:\\Users\\marte\\Downloads\\Well tops.txt"
rokdoc_tops_file = os.path.join(test_file_dir, "RokDocTops.csv")
another_tops_file = os.path.join(test_file_dir, "strat_litho_wellbore.csv")
well_path_file1 = "C:\\Users\\marte\\Downloads\\25_2_6___wellpath.txt"
well_path_file2 = os.path.join(test_file_dir, "Well C wellpath.txt")
checkshot_file = os.path.join(test_file_dir, "Petrel_checkshots.ascii")


class IOTestCase(unittest.TestCase):

    def test_read_file(self):
        data = read_file(las_file1, 'space', 53, {}, {})
        print(data)
        self.assertIsInstance(data, list, 'Test las_file1')

        data = read_file(petrel_well_tops_file, 'space', 17, {}, {})
        print(data)
        self.assertIsInstance(data, list, 'Test petrel_well_tops_file')

        data = read_file(rokdoc_tops_file, ';', 5, {}, {})
        print(data)
        self.assertIsInstance(data, list, 'Test rokdoc_tops_file')

        data = read_file(another_tops_file, ',', 1, {}, {})
        print(data)
        self.assertIsInstance(data, list, 'Test another_tops_file')

        data = read_file(well_path_file1, 'space', 1, {}, {})
        print(data)
        self.assertIsInstance(data, list, 'Test well_path_file1')

        data = read_file(well_path_file2, 'tab', 1, {}, {})
        print(data)
        self.assertIsInstance(data, list, 'Test well_path_file2')

        data = read_file(checkshot_file, 'space', 15, {}, {})
        print(data)
        self.assertIsInstance(data, list, 'Test checkshot_file')



