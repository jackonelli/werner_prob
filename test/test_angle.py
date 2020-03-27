"""Testing angle measure"""
import unittest
import math
import torch
import torch_testing as tt
from src import loss

NUM_PLACES = 5


class TestAngle(unittest.TestCase):
    def test_orthogonal_unit_vector(self):
        x_vec = torch.Tensor([1.0, 0.0])
        y_vec = torch.Tensor([0.0, 1.0])
        true_angle = math.pi / 2
        calc_angle = loss.angle(x_vec, y_vec)
        self.assertAlmostEqual(true_angle,
                               calc_angle.item(),
                               places=NUM_PLACES)

    def test_orthogonal_non_unit_vector(self):
        x_vec = torch.Tensor([1.0, 0.0])
        y_vec = torch.Tensor([0.0, 10.0])
        true_angle = math.pi / 2
        calc_angle = loss.angle(x_vec, y_vec)
        self.assertAlmostEqual(true_angle,
                               calc_angle.item(),
                               places=NUM_PLACES)

    def test_orthogonal_unit_vector(self):
        x_vec = torch.Tensor([-1.0, 0.0])
        y_vec = torch.Tensor([0.0, 1.0])
        true_angle = math.pi / 2
        calc_angle = loss.angle(x_vec, y_vec)
        self.assertAlmostEqual(true_angle,
                               calc_angle.item(),
                               places=NUM_PLACES)

    def test_one_ref_multi_comp(self):
        x_vec = torch.Tensor([1.0, 0.0])
        y_vec = torch.Tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
        true_angle = torch.Tensor([0.0, math.pi / 2, math.pi])
        calc_angle = loss.angle(x_vec, y_vec)
        tt.assert_almost_equal(true_angle, calc_angle)


if __name__ == '__main__':
    unittest.main()
