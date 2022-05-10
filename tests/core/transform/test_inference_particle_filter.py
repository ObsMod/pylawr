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
from unittest.mock import MagicMock

# External modules
import numpy as np
import xarray as xr
import pandas as pd

# Internal modules
from pylawr.transform.inference.particle_filter import SIRParticleFilter
from pylawr.transform.inference.probability import laplace_pdf
from pylawr.transform.inference.predict import random_walk
from pylawr.remap.kernel import gaussian_rbf
from pylawr.grid import PolarGrid
from pylawr.functions.fit import sample_variogram
from pylawr.functions.input import read_lawr_ascii


logging.basicConfig(level=logging.DEBUG)
rnd = np.random.RandomState(42)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


def kernel_variogram(params, obs_dist, **kwargs):   # pragma: no cover
    kernel = gaussian_rbf()
    for i, p_inst in enumerate(kernel.params):
        p_inst.value = params[i]
    variogram = kernel.variogram(obs_dist)
    return variogram


class TestParticleFilter(unittest.TestCase):
    def setUp(self):
        self.ens_size = 100
        self.params_fg = np.repeat(np.array((1, 200))[None, ...],
                                   repeats=self.ens_size, axis=0)
        self.filter = SIRParticleFilter(
            kernel_variogram, random_walk, laplace_pdf,
            ens_size=self.ens_size, params_fg=self.params_fg
        )
        self.source = PolarGrid()
        with open(os.path.join(DATA_PATH, 'lawr_data.txt')) as fh:
            self.array, _ = read_lawr_ascii(fh, self.source)
        self.array = self.array.lawr.set_grid_coordinates(self.source)
        self.obs_dist, self.obs_vario = sample_variogram(self.array, rnd=rnd)

    def test_propagate_uses_predict_func(self):
        propagated = random_walk(self.params_fg)
        self.filter.predict_func = MagicMock(return_value=propagated)
        self.filter._propagate()
        self.filter.predict_func.assert_called_once()
        self.assertEqual(len(self.filter.predict_func.call_args[0]), 1)
        np.testing.assert_equal(self.filter.predict_func.call_args[0][0],
                                self.filter.param_hist[-1])

    def test_propagate_uses_kwargs(self):
        propagated = random_walk(self.params_fg)
        self.filter.predict_func = MagicMock(return_value=propagated)
        self.filter._propagate(noise=0.2)
        self.assertEqual(len(self.filter.predict_func.call_args[0]), 1)
        self.assertEqual(len(self.filter.predict_func.call_args[1]), 1)
        np.testing.assert_equal(self.filter.predict_func.call_args[0][0],
                                self.filter.param_hist[-1])
        self.assertDictEqual(self.filter.predict_func.call_args[1],
                             {'noise': 0.2})

    def test_update_calls_obs_operator(self):
        variogram = kernel_variogram(self.params_fg[0], self.obs_dist)
        self.filter.obs_op_func = MagicMock(return_value=variogram)
        _ = self.filter._update_weights(
            self.params_fg, self.obs_vario, obs_dist=self.obs_dist
        )
        self.assertEqual(self.filter.obs_op_func.call_count, self.ens_size)

    def test_update_calls_prob_func(self):
        self.filter.prob_func = MagicMock(return_value=1)
        _ = self.filter._update_weights(
            self.params_fg, self.obs_vario, obs_dist=self.obs_dist
        )
        self.assertEqual(self.filter.prob_func.call_count, self.ens_size)

    def test_update_returns_normalized_weights(self):
        new_params = self.filter._propagate(noise=0.1)
        new_weights = self.filter._update_weights(
            new_params, self.obs_vario, obs_dist=self.obs_dist
        )
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(new_weights,
                                       self.filter.weight_hist[-1])
        self.assertAlmostEqual(np.sum(new_weights), 1)

    def test_resample_resample_with_given_random_state(self):
        new_params = self.filter._propagate(noise=0.1)
        new_weights = self.filter._update_weights(
            new_params, self.obs_vario, obs_dist=self.obs_dist
        )
        tmp_rnd = np.random.RandomState(42)
        idx = tmp_rnd.choice(self.ens_size, size=self.ens_size, p=new_weights)
        sampled_params = new_params[idx]
        ret_params, ret_weights = self.filter._resample(new_params, new_weights,
                                                        rnd=42)
        np.testing.assert_equal(ret_weights, 0.01)
        np.testing.assert_equal(ret_params, sampled_params)

    def test_fit_calls_propagate(self):
        new_params = self.filter._propagate(noise=0.2)
        self.filter._propagate = MagicMock(return_value=new_params)
        _ = self.filter.fit(obs=self.obs_vario, obs_dist=self.obs_dist,
                            noise=0.2)
        self.filter._propagate.assert_called_once_with(obs_dist=self.obs_dist,
                                                       noise=0.2)

    def test_fit_calls_update(self):
        new_params = self.filter._propagate(noise=0.2)
        new_weights = self.filter._update_weights(new_params, self.obs_vario,
                                                  obs_dist=self.obs_dist)
        self.filter._update_weights = MagicMock(return_value=new_weights)
        _ = self.filter.fit(obs=self.obs_vario, obs_dist=self.obs_dist,
                            noise=0.2)
        self.filter._update_weights.assert_called_once()
        self.assertDictEqual(
            self.filter._update_weights.call_args[1],
            {'obs_dist': self.obs_dist, 'noise': 0.2}
        )

    def test_fit_calls_resample_if_threshold(self):
        new_params = self.filter._propagate(noise=0.2)
        new_weights = self.filter._update_weights(new_params, self.obs_vario,
                                                  obs_dist=self.obs_dist)
        tmp_rnd = np.random.RandomState(42)
        resample_result = self.filter._resample(
            new_params, new_weights, rnd=tmp_rnd
        )
        self.filter._resample = MagicMock(return_value=resample_result)
        self.filter.ens_threshold = 100000000
        _ = self.filter.fit(obs=self.obs_vario, obs_dist=self.obs_dist,
                            noise=0.2, rnd=42)
        self.filter.ens_threshold = 0
        _ = self.filter.fit(obs=self.obs_vario, obs_dist=self.obs_dist,
                            noise=0.2, rnd=42)
        self.filter._resample.assert_called_once()
        self.assertDictEqual(
            self.filter._resample.call_args[1],
            {'obs_dist': self.obs_dist, 'noise': 0.2, 'rnd': 42}
        )

    def test_fit_sets_new_weights_and_params(self):
        new_params = self.filter._propagate(noise=0.2)
        new_weights = self.filter._update_weights(new_params, self.obs_vario,
                                                  obs_dist=self.obs_dist)
        tmp_rnd = np.random.RandomState(42)
        resample_result = self.filter._resample(
            new_params, new_weights, rnd=tmp_rnd
        )
        self.filter._resample = MagicMock(return_value=resample_result)
        self.filter.ens_threshold = 100000000
        _ = self.filter.fit(obs=self.obs_vario, obs_dist=self.obs_dist,
                            noise=0.2, rnd=42)
        np.testing.assert_equal(self.filter.param_hist[-1], resample_result[0])
        np.testing.assert_equal(self.filter.weight_hist[-1], resample_result[1])

    def test_fit_returns_mean_params(self):
        new_params = self.filter._propagate(noise=0.2)
        new_weights = self.filter._update_weights(new_params, self.obs_vario,
                                                  obs_dist=self.obs_dist)
        tmp_rnd = np.random.RandomState(42)
        resample_result = self.filter._resample(
            new_params, new_weights, rnd=tmp_rnd
        )
        max_params = np.average(resample_result[0], axis=0,
                                weights=resample_result[1])
        self.filter._resample = MagicMock(return_value=resample_result)
        self.filter.ens_threshold = 100000000
        ret_params = self.filter.fit(obs=self.obs_vario, obs_dist=self.obs_dist,
                                     noise=0.2, rnd=42)
        np.testing.assert_equal(ret_params, max_params)

    def test_fitted_returns_if_param_hist_larger_one(self):
        self.filter._param_hist = [self.params_fg]
        self.assertFalse(self.filter.fitted)
        self.filter._param_hist = [self.params_fg, self.params_fg]
        self.assertTrue(self.filter.fitted)

    def test_fit_adds_time(self):
        self.assertListEqual(self.filter._times, [pd.NaT])
        _ = self.filter.fit(obs=self.obs_vario, obs_dist=self.obs_dist,
                            noise=0.2, rnd=42)
        self.assertListEqual(self.filter._times, [pd.NaT, pd.NaT])

    def test_fit_adds_time_if_set(self):
        self.assertListEqual(self.filter._times, [pd.NaT])
        _ = self.filter.fit(obs=self.obs_vario, time=0, obs_dist=self.obs_dist,
                            noise=0.2, rnd=42)
        self.assertListEqual(self.filter._times, [pd.NaT, pd.to_datetime(0)])

    def test_to_xarray_sets_time_as_coordinate(self):
        self.filter._times = [pd.to_datetime('25.12.1992 08:00'), ]
        time_coord = pd.DatetimeIndex(self.filter._times)
        time_coord.names = ['time']
        filter_arr = self.filter.to_xarray()
        self.assertIn('time', filter_arr.dims)
        self.assertIn('time', filter_arr.coords)
        pd.testing.assert_index_equal(filter_arr.indexes['time'], time_coord)

    def test_to_xarray_has_parameters(self):
        right_xr = xr.DataArray(
            data=np.array(self.filter.param_hist),
            coords={
                'time': self.filter._times,
                'ensemble': range(self.ens_size)
            },
            dims=['time', 'ensemble', 'param_id']
        )
        right_xr.name = 'parameters'
        pf_ds = self.filter.to_xarray()
        xr.testing.assert_identical(pf_ds['parameters'], right_xr)

    def test_to_xarray_has_weights(self):
        right_xr = xr.DataArray(
            data=np.array(self.filter.weight_hist),
            coords={
                'time': self.filter._times,
                'ensemble': range(self.ens_size)
            },
            dims=['time', 'ensemble']
        )
        right_xr.name = 'weights'
        pf_ds = self.filter.to_xarray()
        xr.testing.assert_identical(pf_ds['weights'], right_xr)


if __name__ == '__main__':
    unittest.main()
