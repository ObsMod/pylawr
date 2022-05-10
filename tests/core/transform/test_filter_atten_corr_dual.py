#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import unittest
import logging
import os

# External modules
import numpy as np
import xarray as xr
from sklearn.isotonic import IsotonicRegression

# Internal modules
from pylawr.grid import PolarGrid, CartesianGrid
from pylawr.transform.attenuation.atten_corr_dual import \
    AttenuationCorrectionDual
from pylawr.utilities.helpers import create_array

logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestAttenuationCorrectionDual(unittest.TestCase):
    def setUp(self):
        self.corrector = AttenuationCorrectionDual()
        self.grid = PolarGrid()
        self.refl_dwd = create_array(self.grid, 30)
        self.atten = np.arange(self.grid.nr_ranges) * 20 / self.grid.nr_ranges
        self.refl_pattern = self.refl_dwd - self.atten
        self.refl_pattern = self.refl_pattern.lawr.set_grid_coordinates(
            self.grid
        )

    def test_fitted_is_false_if_none(self):
        self.corrector._attenuation = None
        self.assertFalse(self.corrector.fitted)
        self.corrector._attenuation = 123
        self.assertTrue(self.corrector.fitted)

    def test_check_grids_checks_if_grids_are_polar_and_equal(self):
        with self.assertRaises(TypeError):
            self.corrector._check_grids(CartesianGrid(), self.grid)
        with self.assertRaises(TypeError):
            self.corrector._check_grids(self.grid, CartesianGrid())
        grid_temp = PolarGrid(center=(1, 1, 1))
        with self.assertRaises(ValueError):
            self.corrector._check_grids(self.grid, grid_temp)

    def test_k_max_returns_difference(self):
        returned_kmax = self.corrector._calc_kmax(self.refl_pattern,
                                                  self.refl_dwd)
        correct_kmax = self.refl_dwd - self.refl_pattern
        xr.testing.assert_identical(returned_kmax, correct_kmax)

    def test_k_max_returnes_like_lengfeld(self):
        pattern_z = self.refl_pattern.lawr.to_z()
        dwd_z = self.refl_dwd.lawr.to_z()
        ratio = dwd_z / pattern_z
        k_max_lengfeld = 10 * np.log10(ratio)
        returned_kmax = self.corrector._calc_kmax(self.refl_pattern,
                                                  self.refl_dwd)
        np.testing.assert_almost_equal(returned_kmax.values,
                                       k_max_lengfeld.values)

    def test_fit_sets_attenuation(self):
        self.assertIsNone(self.corrector._attenuation)
        self.corrector.fit(refl_attenuated=self.refl_pattern,
                           refl_robust=self.refl_dwd)
        self.assertIsInstance(self.corrector._attenuation, xr.DataArray)

    def test_fit_if_regression_is_none_kmax(self):
        self.corrector.fit(refl_attenuated=self.refl_pattern,
                           refl_robust=self.refl_dwd, regression=None)
        right_atten = self.corrector._calc_kmax(self.refl_pattern,
                                                self.refl_dwd)
        xr.testing.assert_equal(self.corrector._attenuation, right_atten)

    def test_calc_attenuation_replaces_smaller_zero_with_replace_neg(self):
        k_max = -self.corrector._calc_kmax(self.refl_pattern, self.refl_dwd)-1
        attenuation = self.corrector._regress_attenuation(
            k_max, replace_neg=-9999
        )
        np.testing.assert_equal(attenuation.values, -9999)

    def test_nan_values_are_masked(self):
        k_max = self.corrector._calc_kmax(self.refl_pattern, self.refl_dwd)
        k_max[:, :, 10] = np.nan
        attenuation = self.corrector._regress_attenuation(
            k_max
        )
        np.testing.assert_equal(attenuation[:, :, 10].values, np.nan)

    def test_attenuation_regress_kmax(self):
        k_max = self.corrector._calc_kmax(self.refl_pattern, self.refl_dwd)
        attenuation = k_max.copy()
        regression = IsotonicRegression()
        for ind in np.ndindex(*k_max.shape[:-1]):
            attenuation[ind] = regression.fit_transform(
                k_max['range'].values.astype(np.float64), k_max.values[ind]
            )
        returned_attenuation = self.corrector._regress_attenuation(
            k_max, regression=regression
        )
        xr.testing.assert_identical(returned_attenuation, attenuation)

    def test_fit_sets_regressed_attenuation(self):
        k_max = self.corrector._calc_kmax(self.refl_pattern, self.refl_dwd)
        regression = IsotonicRegression()
        correct_attenuation = self.corrector._regress_attenuation(
            k_max, regression=regression
        )
        self.corrector.fit(self.refl_pattern, self.refl_dwd,
                           regression=regression)
        xr.testing.assert_equal(self.corrector.attenuation, correct_attenuation)

    def test_to_xarray_returns_dataset(self):
        self.corrector.fit(self.refl_pattern, self.refl_dwd)
        right_ds = self.corrector.attenuation.to_dataset()
        right_ds.attrs['type'] = self.__class__.__name__
        returned_ds = self.corrector.to_xarray()
        xr.testing.assert_equal(returned_ds, right_ds)


if __name__ == '__main__':
    unittest.main()
