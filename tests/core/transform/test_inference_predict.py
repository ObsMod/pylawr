#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2/28/19

Created for pattern

    Copyright (C) {2019}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules
import unittest
import logging
import os

# External modules
import numpy as np

# Internal modules
from pylawr.transform.inference import predict

logging.basicConfig(level=logging.DEBUG)
rnd = np.random.RandomState(42)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestRandomWalk(unittest.TestCase):
    def setUp(self):
        self.parameters = rnd.uniform(0, 10000, size=(5000, 3))

    def test_random_walk_stddev_noise(self):
        new_params = predict.random_walk(self.parameters)
        param_delta = new_params - self.parameters
        norm_delta = param_delta / self.parameters
        self.assertAlmostEqual(np.mean(norm_delta), 0, places=2)
        self.assertAlmostEqual(np.std(norm_delta), 0.1, places=2)
        logging.debug('Mean: {0:.4f}'.format(np.mean(norm_delta)))
        logging.debug('StdDev: {0:.4f}'.format(np.std(norm_delta)))

    def test_random_walk_clips_delta(self):
        new_params = predict.random_walk(self.parameters)
        param_delta = new_params - self.parameters
        upper_clip = np.all(param_delta < self.parameters)
        lower_clip = np.all(param_delta > -self.parameters)
        clipped = np.logical_and(upper_clip, lower_clip)
        self.assertTrue(clipped)

    def test_random_walk_noise_percentage(self):
        new_params = predict.random_walk(self.parameters, noise=0.05)
        param_delta = new_params - self.parameters
        norm_delta = param_delta / self.parameters
        self.assertAlmostEqual(np.std(norm_delta), 0.05, places=2)
        logging.debug('StdDev: {0:.4f}'.format(np.std(norm_delta)))


if __name__ == '__main__':
    unittest.main()
