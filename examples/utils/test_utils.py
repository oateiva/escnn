import unittest
import torch
from examples.utils.example_utils import se2_geodesic_loss, se2_chordal_loss

class TestSE2GeodesicLoss(unittest.TestCase):

    def test_se2_geodesic_loss(self):
        # Define test cases
        T_pred = torch.tensor([
            [[1.0, 0.0, 1.0],
             [0.0, 1.0, 2.0],
             [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 3.0],
             [1.0, 0.0, 4.0],
             [0.0, 0.0, 1.0]]
        ])

        T_true = torch.tensor([
            [[1.0, 0.0, 1.0],
             [0.0, 1.0, 2.0],
             [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 3.0],
             [1.0, 0.0, 4.0],
             [0.0, 0.0, 1.0]]
        ])

        # Expected results
        expected_d_T = 0.0
        expected_d_R = 0.0

        # Compute geodesic loss
        d_T, d_R = se2_chordal_loss(T_pred, T_true)

        # Assert the results
        self.assertAlmostEqual(d_T.item(), expected_d_T, places=6)
        self.assertAlmostEqual(d_R.item(), expected_d_R, places=6)

if __name__ == '__main__':
    unittest.main()