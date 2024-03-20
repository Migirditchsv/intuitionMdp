import unittest
import numpy as np
from src.mfpt import get_linear_index, linear_index_to_2D, get_ranked_states

class TestGetRankedStates(unittest.TestCase):
    def test_get_ranked_states(self):
        # Define a known input and expected output
        input_matrix = np.array([[3, 1, 2], [6, 4, 5], [9, 7, 8]])
        expected_output = [(0, 1), (0, 2), (0, 0), (1, 1), (1, 2), (1, 0), (2, 1), (2, 2), (2, 0)]

        # Call the function with the known input
        output = get_ranked_states(input_matrix)

        # Assert that the output is as expected
        self.assertEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()

class TestArrayFunctions(unittest.TestCase):
    def setUp(self):
        self.array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_get_linear_index(self):
        self.assertEqual(get_linear_index((1, 0), self.array_2d), 3)

    def test_reshape_to_2d(self):
        reshaped_array = linear_index_to_2D(self.array_1d, self.array_2d)
        np.testing.assert_array_equal(reshaped_array, self.array_2d)

    def test_consistency(self):
        reshaped_array = linear_index_to_2D(self.array_1d, self.array_2d)
        for i in range(self.array_2d.shape[0]):
            for j in range(self.array_2d.shape[1]):
                linear_index = get_linear_index((i, j), self.array_2d)
                self.assertEqual(reshaped_array.flat[linear_index], self.array_2d[i, j])

if __name__ == '__main__':
    unittest.main()