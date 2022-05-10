#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch, PropertyMock

# External modules
import matplotlib.figure as mpl_figure
import matplotlib.collections as mpl_collections

import xarray as xr
import numpy as np
import cartopy.crs as ccrs

# Internal modules
from pylawr.grid import PolarGrid, LatLonGrid
from pylawr.plot.layer.radarfield import RadarFieldLayer
from pylawr.utilities.helpers import create_array


logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

rnd = np.random.RandomState(42)


BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestClass(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()
        self.array = create_array(self.grid)
        self.figure = mpl_figure.Figure()
        self.axes = self.figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.layer = RadarFieldLayer(self.array, grid=self.grid)

    def test_grid_returns_grid_if_set(self):
        self.layer._grid = self.grid
        self.assertEqual(self.layer.grid, self.layer._grid)
        self.layer._grid = LatLonGrid()
        self.assertEqual(self.layer.grid, self.layer._grid)

    def test_grid_returns_none_if_not_set(self):
        self.layer._grid = None
        self.assertIsNone(self.layer.grid)

    def test_grid_sets_private_grid(self):
        self.assertEqual(self.layer._grid, self.grid)
        set_grid = LatLonGrid()
        self.layer.grid = set_grid
        self.assertNotEqual(self.layer._grid, self.grid)
        self.assertEqual(self.layer._grid, set_grid)

    def test_grid_raises_type_error_if_not_none_or_grid(self):
        self.layer.grid = None
        self.layer.grid = self.grid
        with self.assertRaises(TypeError):
            self.layer.grid = [self.grid, ]

    def test_field_returns_private_field(self):
        self.layer._field = None
        self.assertIsNone(self.layer.field)
        self.layer._field = self.array
        xr.testing.assert_identical(self.array, self.layer.field)

    def test_field_sets_private_field(self):
        self.layer._field = None
        self.layer.field = self.array
        self.assertIsNotNone(self.layer._field)
        xr.testing.assert_identical(self.layer._field, self.array)

    def test_field_raises_type_error_if_no_xr_array(self):
        self.layer.field = self.array
        self.layer.field = self.array.values
        with self.assertRaises(TypeError):
            self.layer.field = 123

    def test_field_raises_value_error_if_not_castable(self):
        tmp_array = xr.concat([self.array, self.array], dim='time')
        with self.assertRaises(ValueError):
            self.layer.field = tmp_array

    def test_get_checked_grid_returns_set_grid(self):
        self.layer.grid = self.grid
        checked_grid = self.layer._get_checked_grid()
        self.assertEqual(self.grid, checked_grid)

    def test_get_checked_grid_returns_field_grid_if_none(self):
        self.layer.grid = None
        self.layer.field.lawr.grid = self.grid
        checked_grid = self.layer._get_checked_grid()
        self.assertEqual(self.grid, checked_grid)

    def test_get_checked_grid_raises_attribute_error_if_not_available(self):
        self.layer.grid = None
        self.layer.field.lawr.grid = None
        with self.assertRaises(AttributeError):
            _ = self.layer._get_checked_grid()

    def test_get_checked_grid_raises_attribute_error_if_numpy(self):
        self.layer.field = self.layer.field.values
        self.layer.grid = None
        with self.assertRaises(AttributeError):
            _ = self.layer._get_checked_grid()

    def test_get_checked_grid_skips_checks_if_numpy(self):
        self.layer.field = self.layer.field.values
        self.layer.grid = self.grid
        grid = self.layer._get_checked_grid()
        self.assertEqual(grid, self.grid)

    @patch('pylawr.field.RadarField.check_grid')
    def test_get_grid_checks_grid(self, p):
        self.layer.grid = LatLonGrid()
        _ = self.layer._get_checked_grid()
        p.assert_called_once_with(grid=self.layer.grid)

    def test_plot_gets_checked_grid(self):
        trg = 'pylawr.plot.layer.radarfield.RadarFieldLayer._get_checked_grid'
        with patch(trg, return_value=self.grid) as p:
            self.layer.plot(ax=self.axes)
            p.assert_called_once_with()

    def test_plot_calls_lat_lon_bounds_from_checked_grid(self):
        lat_bounds, lon_bounds = self.grid.lat_lon_bounds
        with patch('pylawr.grid.base.BaseGrid.lat_lon_bounds',
                   new_callable=PropertyMock) as p:
            p.return_value = lat_bounds, lon_bounds
            self.layer.plot(ax=self.axes)
            p.assert_called_once()

    def test_plot_masks_array_data(self):
        data = self.array.values.squeeze()
        mask = ~np.isfinite(data)
        with patch('numpy.ma.array', return_value=data) as p:
            self.layer.plot(ax=self.axes)
            p.assert_called_once()
        np.testing.assert_equal(data, p.call_args_list[0][1]['data'])
        np.testing.assert_equal(mask, p.call_args_list[0][1]['mask'])

    def test_plot_calls_pcolormesh(self):
        self.layer.settings = {}
        data = self.array.values.squeeze()
        mask = ~np.isfinite(data)
        masked_data = np.ma.array(data, mask=mask)
        lat_bounds, lon_bounds = self.grid.lat_lon_bounds
        plot_store = self.axes.pcolormesh(
            lon_bounds, lat_bounds, masked_data,
            transform=self.layer._transform, zorder=self.layer.zorder
        )
        with patch('cartopy.mpl.geoaxes.GeoAxes.pcolormesh',
                   return_value=plot_store) as p:
            self.layer.plot(ax=self.axes)
            p.assert_called_once()
        np.testing.assert_equal(p.call_args_list[0][0][0], lon_bounds)
        np.testing.assert_equal(p.call_args_list[0][0][1], lat_bounds)
        np.testing.assert_equal(p.call_args_list[0][0][2], masked_data)
        self.assertEqual(p.call_args_list[0][1]['transform'],
                         self.layer._transform)
        self.assertEqual(p.call_args_list[0][1]['zorder'], self.layer.zorder)

    def test_plot_calls_pcolormesh_with_settings(self):
        self.layer.settings = dict(cmap='grey', alpha=0.5)
        data = self.array.values.squeeze()
        mask = ~np.isfinite(data)
        masked_data = np.ma.array(data, mask=mask)
        lat_bounds, lon_bounds = self.grid.lat_lon_bounds
        plot_store = self.axes.pcolormesh(
            lon_bounds, lat_bounds, masked_data,
            transform=self.layer._transform, zorder=self.layer.zorder
        )
        with patch('cartopy.mpl.geoaxes.GeoAxes.pcolormesh',
                   return_value=plot_store) as p:
            self.layer.plot(ax=self.axes)
            p.assert_called_once()
        self.assertEqual(p.call_args_list[0][1]['cmap'], 'grey')
        self.assertEqual(p.call_args_list[0][1]['alpha'], 0.5)

    def test_plot_sets_plot_store_to_returned_values(self):
        self.assertIsNone(self.layer.plot_store)
        self.layer.plot(ax=self.axes)
        self.assertIsNotNone(self.layer.plot_store)
        self.assertIsInstance(self.layer.plot_store, mpl_collections.QuadMesh)

    def test_plot_appends_plot_store_to_collection(self):
        self.assertFalse(self.layer.collection)
        self.layer.plot(ax=self.axes)
        self.assertTrue(self.layer.collection)
        self.assertListEqual([self.layer.plot_store, ], self.layer.collection)
        self.layer.plot(ax=self.axes)
        self.assertEqual(2, len(self.layer.collection))

    def test_plot_works_with_numpy_data(self):
        self.layer.field = self.layer.field.values
        self.layer.plot(self.axes)


if __name__ == '__main__':
    unittest.main()
