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
import scipy.stats

# Internal modules
from pylawr.transform.inference import probability

logging.basicConfig(level=logging.DEBUG)

rnd = np.random.RandomState(42)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestGaussian(unittest.TestCase):
    def setUp(self):
        self.y_obs = rnd.normal(size=(1000, ))
        self.y_hat = self.y_obs + 0.1

    def test_single_val(self):
        prob = scipy.stats.norm.pdf(0.5, loc=0, scale=1)
        nll = probability.gaussian_pdf(0, 0.5, 1)
        ret_prob = np.exp(nll)
        self.assertAlmostEqual(prob, ret_prob)

    def test_returns_exp_of_squared_err(self):
        nll = np.sum(
            np.log([scipy.stats.norm.pdf(self.y_obs[i], loc=loc)
             for i, loc in enumerate(self.y_hat)])
        )
        ret_nll = probability.gaussian_pdf(self.y_hat, self.y_obs)
        np.testing.assert_almost_equal(ret_nll, nll)

    def test_float_var_is_used(self):
        var = 2
        nll = np.sum(
            np.log([scipy.stats.norm.pdf(self.y_obs[i], loc=loc, scale=np.sqrt(var))
             for i, loc in enumerate(self.y_hat)])
        )
        ret_nll = probability.gaussian_pdf(self.y_hat, self.y_obs, var=var)
        np.testing.assert_almost_equal(ret_nll, nll)

    def test_array_var_is_used(self):
        var = rnd.uniform(1, 5, size=(1000, ))
        nll = np.sum(
            np.log([scipy.stats.norm.pdf(self.y_obs[i], loc=loc, scale=np.sqrt(var[i]))
             for i, loc in enumerate(self.y_hat)])
        )
        ret_nll = probability.gaussian_pdf(self.y_hat, self.y_obs, var=var)
        np.testing.assert_almost_equal(ret_nll, nll)


class TestLaplace(unittest.TestCase):
    def setUp(self):
        self.y_obs = rnd.normal(size=(1000, ))
        self.y_hat = self.y_obs + 0.1

    def test_single_val(self):
        scale = np.sqrt(1/2)
        prob = scipy.stats.laplace.pdf(0.5, loc=0, scale=scale)
        ret_nll = probability.laplace_pdf(0, 0.5, 1)
        ret_prob = np.exp(ret_nll)
        self.assertAlmostEqual(prob, ret_prob)

    def test_returns_exp_of_squared_err(self):
        scale = np.sqrt(1/2)
        nll = np.sum(
            np.log([scipy.stats.laplace.pdf(self.y_obs[i], loc=loc, scale=scale)
             for i, loc in enumerate(self.y_hat)])
        )
        ret_nll = probability.laplace_pdf(self.y_hat, self.y_obs)
        np.testing.assert_almost_equal(ret_nll, nll)

    def test_float_var_is_used(self):
        var = 2
        scale = np.sqrt(2/2)
        nll = np.sum(
            np.log([scipy.stats.laplace.pdf(self.y_obs[i], loc=loc, scale=scale)
             for i, loc in enumerate(self.y_hat)])
        )
        ret_nll = probability.laplace_pdf(self.y_hat, self.y_obs, var=var)
        np.testing.assert_almost_equal(ret_nll, nll)

    def test_array_var_is_used(self):
        var = rnd.uniform(1, 5, size=(1000, ))
        scale = np.sqrt(var/2)
        nll = np.sum(
            np.log([scipy.stats.laplace.pdf(self.y_obs[i], loc=loc, scale=scale[i])
             for i, loc in enumerate(self.y_hat)])
        )
        ret_nll = probability.laplace_pdf(self.y_hat, self.y_obs, var=var)
        np.testing.assert_almost_equal(ret_nll, nll)


if __name__ == '__main__':
    unittest.main()
