#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os

# External modules
import numpy

# Internal modules
from pylawr.remap.kernel.base_ops import KernelNode
from pylawr.remap.kernel.kernels import *


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestGPKernels(unittest.TestCase):
    def setUp(self):
        self.distances = np.random.normal(scale=10000, size=(100, 100))

    def test_gaussian_rbf(self):
        length_scale = 4000
        stddev = 5
        dist_squared = self.distances ** 2
        inner_prod = -dist_squared / (2*length_scale**2)
        exp_prod = np.exp(inner_prod)
        rbf_val = stddev ** 2 * exp_prod
        ret_kernel = gaussian_rbf(length_scale=length_scale, stddev=stddev)
        self.assertIsInstance(ret_kernel, KernelNode)
        ret_val = ret_kernel(self.distances)
        np.testing.assert_almost_equal(rbf_val, ret_val)

    def test_variogram_returns_variogram_values(self):
        length_scale = 4000
        stddev = 5
        ret_kernel = gaussian_rbf(length_scale=length_scale, stddev=stddev)
        distances = np.linspace(0, 10000, 100)
        variogram = ret_kernel.diag(np.zeros_like(distances))
        variogram -= ret_kernel.diag(distances)
        ret_variogram = ret_kernel.variogram(distances)
        np.testing.assert_almost_equal(ret_variogram, variogram)

    def test_exp_sin_squared(self):
        periodicity = np.pi * 2
        length_scale = 10000

        arg = np.pi * self.distances / periodicity
        sin_of_arg = np.sin(arg)
        right_val = np.exp(- 2 * (sin_of_arg / length_scale) ** 2)
        ret_kernel = exp_sin_squared(length_scale=length_scale,
                                     periodicity=periodicity)
        self.assertIsInstance(ret_kernel, KernelNode)
        ret_val = ret_kernel(self.distances)
        np.testing.assert_almost_equal(right_val, ret_val)

    def test_rational_quadratic(self):
        length_scale = 4000
        stddev = 5
        scale = 4

        inner_kernel = 1 + self.distances ** 2 / (2 * scale * length_scale)
        scaled_kernel = inner_kernel ** (-scale)
        right_val = stddev ** 2 * scaled_kernel

        ret_kernel = rational_quadratic(
            length_scale=length_scale, stddev=stddev, scale=scale
        )
        self.assertIsInstance(ret_kernel, KernelNode)
        ret_val = ret_kernel(self.distances)
        np.testing.assert_almost_equal(right_val, ret_val)


if __name__ == '__main__':
    unittest.main()
