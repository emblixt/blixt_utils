import unittest
import numpy as np
import matplotlib.pyplot as plt


class TestCase(unittest.TestCase):

    def test_add_one(self):
        from blixt_utils.utils import add_one
        strings = ['A', 'A1', 'A 1', 'A2', 'A 3']
        expected_results = ['A 1', 'A2', 'A 2', 'A3', 'A 4']
        for i, string in enumerate(strings):
            out = add_one(string)
            print(out)
            self.assertTrue(out == expected_results[i])

    def test_find_value(self):
        from blixt_utils.utils import find_value
        rad = np.linspace(-1 * np.pi, np.pi, 100)
        # x = np.sin(2. * rad) + np.random.rand(100) / 4.
        starting_point = 33
        x = np.sin(4. * rad)
        plt.plot(rad, x)
        plt.scatter(rad[starting_point], x[starting_point], marker='o', c='b')
        val, i = find_value(x, starting_point, snap_to='nearest_max')
        plt.scatter(rad[i], val, marker='^', c='b')
        val, i = find_value(x, starting_point, snap_to='nearest_min')
        plt.scatter(rad[i], val, marker='v', c='b')

        starting_point = 50
        plt.scatter(rad[starting_point], x[starting_point], marker='o', c='g')
        val, i = find_value(x, starting_point, snap_to='nearest_extreme')
        plt.scatter(rad[i], val, marker='v', c='g')

        starting_point = 65
        plt.scatter(rad[starting_point], x[starting_point], marker='o', c='r')
        val, i = find_value(x, starting_point, snap_to='nearest_extreme')
        plt.scatter(rad[i], val, marker='v', c='r')

        plt.show()
        self.assertTrue(True)


