#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
import datetime
from unittest.mock import patch

# External modules
import numpy as np

import xarray as xr

# Internal modules
from pylawr.grid.cartesian import CartesianGrid
from pylawr.grid.polar import PolarGrid
from pylawr.transform.spatial.beamexpansion import TAG_BEAM_EXPANSION_CORR
from pylawr.functions.fit import fit_extrapolator
from pylawr.transform.temporal.extrapolation import Extrapolator
from pylawr.remap.nearest import NearestNeighbor
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestFitFunctions(unittest.TestCase):
    def setUp(self):
        # cartesian
        self.save_path = 'test_array.nc'
        len_x = 300
        len_y = 300
        self.grid = CartesianGrid(nr_points=(len_x, len_y))
        self.array = create_array(self.grid)
        self.array.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
        self.array.to_netcdf('test_array.nc')

        # polar
        self.save_path_polar = 'test_polar_array.nc'
        self.array_polar = create_array(PolarGrid())
        self.array_polar.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
        self.array_polar.to_netcdf('test_polar_array.nc')

    def tearDown(self):
        os.remove(self.save_path)
        os.remove(self.save_path_polar)

    def test_fit_polar_array(self):
        returned_value = fit_extrapolator(refl_array=self.array_polar,
                                          pre_refl_path=self.save_path_polar,
                                          grid_extrapolation=CartesianGrid(),
                                          remapper=NearestNeighbor(1))
        self.assertIsInstance(returned_value, Extrapolator)

    def test_fit_extrapolator_returns_extrapolator(self):
        returned_value = fit_extrapolator(refl_array=self.array,
                                          pre_refl_path=self.save_path,
                                          grid_extrapolation=self.grid,
                                          remapper=NearestNeighbor(1))
        self.assertIsInstance(returned_value, Extrapolator)

    def test_fit_extrapolator_returns_fitted_extrapolator(self):
        returned_value = fit_extrapolator(refl_array=self.array,
                                          pre_refl_path=self.save_path,
                                          grid_extrapolation=self.grid,
                                          remapper=NearestNeighbor(1))
        self.assertTrue(returned_value.fitted)

    @patch('pylawr.transform.temporal.extrapolation.Extrapolator.fit')
    def test_fit_extrapolator_passes_grid_to_fit(self, grid_mock):
        _ = fit_extrapolator(refl_array=self.array,
                             pre_refl_path=self.save_path,
                             grid_extrapolation=self.grid,
                             remapper=NearestNeighbor(1))
        self.assertEqual(id(grid_mock.call_args[1]['grid']), id(self.grid))

    def test_fit_extrapolator_passes_args_kwargs_to_extrapolator(self):
        extrapolator = fit_extrapolator(self.array, self.save_path,
                                        grid_extrapolation=self.grid,
                                        remapper=NearestNeighbor(1),
                                        cut_percentage=0.3,
                                        max_timediff=42)
        self.assertEqual(extrapolator.cut_percentage, 0.3)
        self.assertEqual(extrapolator.max_timediff, 42)

    def test_fit_extrapolator_array_needs_grid(self):
        self.array.lawr.grid = None
        with self.assertRaises(AttributeError) as e:
            _ = fit_extrapolator(self.array, self.save_path, grid=self.grid)


if __name__ == '__main__':
    unittest.main()
