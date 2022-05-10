#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
import tempfile

# External modules
import xarray as xr

import numpy as np

# Internal modules
import pylawr.functions.input as input_funcs
from pylawr.datahandler.lawr import LawrHandler
from pylawr.datahandler.hdf5 import DWDHDF5Handler
from pylawr.grid.polar import PolarGrid
from pylawr.transform.spatial.beamexpansion import TAG_BEAM_EXPANSION_CORR
from pylawr.utilities.conventions import naming_convention
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestInputFunctions(unittest.TestCase):
    def test_read_lawr_txt_returns_polar_grid_if_no_given(self):
        file_path = os.path.join(DATA_PATH, 'lawr_data.txt')
        with open(file_path, 'r') as fh:
            _, returned_grid = input_funcs.read_lawr_ascii(fh, None)
        self.assertIsInstance(returned_grid, PolarGrid)

    def test_read_lawr_returns_read_data(self):
        grid = PolarGrid()
        file_path = os.path.join(DATA_PATH, 'lawr_data.txt')
        with open(file_path, 'r') as fh:
            data_handler = LawrHandler(fh)
            right_refl = data_handler.get_reflectivity()
            right_refl = right_refl.lawr.set_grid_coordinates(grid)
            right_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
            returned_refl, _ = input_funcs.read_lawr_ascii(fh, None)
        xr.testing.assert_identical(returned_refl, right_refl)

    def test_read_lawr_sets_grid_coordinates(self):
        grid = PolarGrid(range_res=420)
        file_path = os.path.join(DATA_PATH, 'lawr_data.txt')
        with open(file_path, 'r') as fh:
            data_handler = LawrHandler(fh)
            right_refl = data_handler.get_reflectivity()
            right_refl = right_refl.lawr.set_grid_coordinates(grid)
            right_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
            returned_refl, _ = input_funcs.read_lawr_ascii(fh, grid=grid)
        xr.testing.assert_identical(returned_refl, right_refl)

    def test_read_lawr_returns_da_following_conventions(self):
        file_path = os.path.join(DATA_PATH, 'lawr_data.txt')
        with open(file_path, 'r') as fh:
            returned_refl, _ = input_funcs.read_lawr_ascii(fh, None)
        self.assertTrue(naming_convention[returned_refl.name].items() <=
                        returned_refl.attrs.items())
        for dim in returned_refl.dims:
            self.assertTrue(naming_convention[dim].items()
                            <= returned_refl[dim].attrs.items())

    def test_read_lawr_returns_given_grid(self):
        grid = PolarGrid(range_res=420)
        file_path = os.path.join(DATA_PATH, 'lawr_data.txt')
        with open(file_path, 'r') as fh:
            _, returned_grid = input_funcs.read_lawr_ascii(fh, grid=grid)
        self.assertEqual(id(grid), id(returned_grid))

    def test_read_dwd_hdf5_returns_read_data(self):
        file_path = os.path.join(DATA_PATH, 'dwd_test_data.hdf5')
        with open(file_path, 'rb') as fh:
            radar_dh = DWDHDF5Handler(fh)
            right_refl = radar_dh.get_reflectivity()
            grid = radar_dh.grid
            right_refl = right_refl.lawr.set_grid_coordinates(grid)
            right_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
            returned_refl, _ = input_funcs.read_dwd_hdf5(fh, grid=grid)
        xr.testing.assert_identical(returned_refl, right_refl)

    def test_read_dwd_hdf5_returns_given_grid(self):
        file_path = os.path.join(DATA_PATH, 'dwd_test_data.hdf5')
        with open(file_path, 'rb') as fh:
            radar_dh = DWDHDF5Handler(fh)
            grid = radar_dh.grid
            _, returned_grid = input_funcs.read_dwd_hdf5(fh, grid=grid)
        self.assertEqual(id(grid), id(returned_grid))

    def test_read_dwd_hdf5_returns_read_grid_if_no_grid_given(self):
        file_path = os.path.join(DATA_PATH, 'dwd_test_data.hdf5')
        with open(file_path, 'rb') as fh:
            radar_dh = DWDHDF5Handler(fh)
            grid = radar_dh.grid
            _, returned_grid = input_funcs.read_dwd_hdf5(fh, grid=None)
        np.testing.assert_equal(
            np.concatenate(grid.coords),
            np.concatenate(returned_grid.coords),
        )

    def test_read_dwd_hdf5_sets_grid_coordinates(self):
        file_path = os.path.join(DATA_PATH, 'dwd_test_data.hdf5')
        with open(file_path, 'rb') as fh:
            radar_dh = DWDHDF5Handler(fh)
            read_refl = radar_dh.get_reflectivity()
            grid = radar_dh.grid
            new_grid = PolarGrid(nr_azi=grid.nr_azi, nr_ranges=grid.nr_ranges)
            wrong_refl = read_refl.lawr.set_grid_coordinates(grid)
            right_refl = read_refl.lawr.set_grid_coordinates(new_grid)
            wrong_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
            right_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
            returned_refl, _ = input_funcs.read_dwd_hdf5(fh, grid=new_grid)
        xr.testing.assert_identical(returned_refl, right_refl)
        try:
            xr.testing.assert_identical(returned_refl, wrong_refl)
            raise ValueError('Gridded data doesn\'t equal!')
        except AssertionError as e:
            pass

    def test_read_dwd_hdf5_returns_da_following_conventions(self):
        file_path = os.path.join(DATA_PATH, 'dwd_test_data.hdf5')
        with open(file_path, 'rb') as fh:
            returned_refl, _ = input_funcs.read_dwd_hdf5(fh)
        self.assertTrue(naming_convention[returned_refl.name].items() <=
                        returned_refl.attrs.items())
        for dim in returned_refl.dims:
            self.assertTrue(naming_convention[dim].items()
                            <= returned_refl[dim].attrs.items())

    def test_read_netcdf_new_returns_opened_netcdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file_path = os.path.join(tmp, 'lawr_data_test.nc')
            tmp_array = create_array(PolarGrid())
            tmp_array.to_netcdf(tmp_file_path)
            returned_refl, _ = input_funcs.read_lawr_nc_new(tmp_file_path,
                                                            None)
            with xr.open_dataset(tmp_file_path) as radar_ds:
                xr.testing.assert_identical(returned_refl, radar_ds['dbz'])

    def test_read_netecdf_returns_grid_if_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file_path = os.path.join(tmp, 'lawr_data_test.nc')
            grid = PolarGrid(range_res=120)
            tmp_array = create_array(grid)
            tmp_array.to_netcdf(tmp_file_path)
            _, returned_grid = input_funcs.read_lawr_nc_new(tmp_file_path,
                                                            grid)
            self.assertEqual(id(grid), id(returned_grid))

    def test_read_netecdf_sets_new_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file_path = os.path.join(tmp, 'lawr_data_test.nc')
            grid = PolarGrid(range_res=120)
            tmp_array = create_array(grid)
            tmp_array.to_netcdf(tmp_file_path)
            with xr.open_dataarray(tmp_file_path) as refl:
                refl = refl.lawr.set_grid_coordinates(grid)
            returned_refl, _ = input_funcs.read_lawr_nc_new(tmp_file_path,
                                                            grid)
            xr.testing.assert_identical(returned_refl, refl)

    def test_read_lawr_nc_level0(self):
        file_path = os.path.join(DATA_PATH, 'lawr_l0_example.nc')
        returned_refl, grid = input_funcs.read_lawr_nc_level0(file_path)
        self.assertEqual(returned_refl.attrs['standard_name'],
                         naming_convention['dbz']['standard_name'])


if __name__ == '__main__':
    unittest.main()
