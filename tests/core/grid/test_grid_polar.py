#!/bin/env python
# -*- coding: UTF-8 -*-

# System modules
import unittest
import logging
import os

# External modules
import numpy as np
import pandas as pd
import xarray as xr

# Internal modules
from pylawr.grid.polar import PolarGrid
from pylawr.utilities.conventions import naming_convention


logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestPolarGrid(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()

    def test_beam_elevation_returns_private(self):
        beam_elevation = 10
        self.grid._beam_elevation = beam_elevation
        self.assertEqual(self.grid.beam_elevation, beam_elevation)

    def test_beam_elevation_setter_sets_private(self):
        beam_elevation = 10.
        self.grid._beam_elevation = None
        self.grid.beam_elevation = beam_elevation
        self.assertAlmostEqual(self.grid._beam_elevation, beam_elevation)

    def test_beam_elevation_raises_typeerror_if_not_float(self):
        with self.assertRaises(TypeError):
            self.grid.beam_elevation = 'bla'

    def test_beam_elevation_get_normalized_float(self):
        self.grid.beam_elevation = 130
        self.assertEqual(self.grid.beam_elevation, 50)
        self.grid.beam_elevation = 50
        self.assertEqual(self.grid.beam_elevation, 50)

    def test_calc_coords_calculate_azimuth_and_range(self):
        self.grid.nr_ranges = 50
        self.grid.nr_azi = 36
        self.grid.range_res = 2800
        azimuth = np.linspace(0, 360, self.grid.nr_azi, endpoint=False) + 5.0
        azimuth = azimuth % 360
        range_dist = np.arange(self.grid.nr_ranges) * self.grid.range_res + \
            self.grid.range_res/2
        returned_coords = self.grid._calc_coordinates()
        np.testing.assert_equal(azimuth, returned_coords[0])
        np.testing.assert_equal(range_dist, returned_coords[1])

    def test_ranges_returns_private_coordinate(self):
        np.testing.assert_equal(self.grid.ranges, self.grid.coords[1])

    def test_azimuth_returns_private_coordinate(self):
        np.testing.assert_equal(self.grid.azimuth, self.grid.coords[0])

    def test_beam_elevation_field_returns_repeated_beam_elevation_float(self):
        self.grid._data_shape = (10, 20)
        self.grid._beam_elevation = 10
        returned_field = self.grid.beam_elevation_field
        data_size = np.product(self.grid._data_shape)
        repeat_times = data_size/1
        repeated_f = np.array(self.grid._beam_elevation).repeat(repeat_times)
        repeated_f = repeated_f.reshape(self.grid._data_shape)
        np.testing.assert_equal(returned_field, repeated_f)

    def test_beam_elevation_field_returns_repeated_beam_elevation_array(self):
        self.grid._beam_elevation = np.arange(self.grid.nr_azi)
        returned_field = self.grid.beam_elevation_field
        repeat_times = self.grid.nr_ranges
        repeated_f = np.array(self.grid._beam_elevation).repeat(repeat_times)
        repeated_f = repeated_f.reshape(self.grid._data_shape)
        np.testing.assert_equal(returned_field, repeated_f)

    def test_beam_elevation_field_repeated_along_ranges_array(self):
        self.grid._beam_elevation = np.arange(self.grid.nr_azi)
        returned_field = self.grid.beam_elevation_field
        along_beam = returned_field[0, :]
        self.assertTrue(np.all(along_beam == along_beam[0]))
        np.testing.assert_equal(self.grid._beam_elevation, returned_field[:, 0])

    def test_calc_altitude_sets_altitude_to_center_height(self):
        re = self.grid.earth_radius
        ke = 4. / 3.

        altitude = (
                np.sqrt(
                    self.grid.center_distance * self.grid.center_distance
                    + ke * ke * re * re +
                    2. * self.grid.center_distance * ke * re *
                    np.sin(np.deg2rad(self.grid.beam_elevation))
                ) - (ke * re) + self.grid.center[2]
        )

        np.testing.assert_almost_equal(
            self.grid.altitude,
            altitude,
            decimal=3)

    def test_coord_fields_returns_meshgrid_field(self):
        x, y = np.meshgrid(*self.grid.coords)
        np.testing.assert_equal(
            np.array(self.grid.coord_fields),
            np.array((x.T, y.T))
        )

    def test_calc_coords2latlon_returns_lat_lon(self):
        returned_coords = self.grid._coords2latlon(*self.grid.coord_fields)
        self.assertEqual(np.round(self.grid.center[0]),
                         np.round(returned_coords[0][0][0]))
        self.assertEqual(np.round(self.grid.center[1]),
                         np.round(returned_coords[1][0][0]))


    def test_get_coords_returns_valid_xarray_coords(self):
        coords = self.grid.get_coordinates()
        self.assertIsInstance(coords, dict)
        self.assertTrue(np.all([(k, ) == v[0] for k, v in coords.items()]))
        self.assertTrue(np.all(
            [isinstance(v[1], np.ndarray) for v in coords.values()]))

    def test_get_coords_returns_xarray_coord_attrs(self):
        coords = self.grid.get_coordinates()
        [self.assertDictEqual(naming_convention[k],
                              v[2]) for k, v in coords.items()]

    def test_get_altitude_returns_xarray_dataarray(self):
        name_da = 'zsl'
        altitude_da = xr.DataArray(
            data=self.grid.altitude,
            coords=self.grid.get_coordinates(),
            dims=self.grid.coord_names,
            name=name_da,
            attrs=naming_convention[name_da]
        )
        xr.testing.assert_identical(self.grid.get_altitude(), altitude_da)

    def test_get_lat_lon_returns_xarray_dataset(self):
        latlon = self.grid.lat_lon
        latlon_ds = xr.Dataset(
            data_vars=dict(
                lat=(self.grid.coord_names, latlon[0],
                     naming_convention['lat']),
                lon=(self.grid.coord_names, latlon[1],
                     naming_convention['lon']),
            ),
            coords=self.grid.get_coordinates()
        )
        xr.testing.assert_identical(self.grid.get_lat_lon(), latlon_ds,)

    def test_range_offset_sets_starts_point_of_ranges(self):
        self.grid.range_offset = -5000
        right_range = np.arange(self.grid.nr_ranges) * self.grid.range_res + \
            self.grid.range_offset + self.grid.range_res / 2
        np.testing.assert_almost_equal(right_range, self.grid.ranges)

    def test_azimuth_offset_rotates_azimuth_angles(self):
        self.grid.azi_offset = 0.3
        right_angles = np.linspace(0, 360, self.grid.nr_azi, endpoint=False,
                                   dtype=np.float64) + \
            self.grid.azi_offset+0.5
        np.testing.assert_equal(self.grid.azimuth, right_angles)

    def test_calc_coordinates_normalize_azimuth_angle(self):
        self.grid.azi_offset = 5.3
        unnormalized = np.linspace(0, 360, self.grid.nr_azi, endpoint=False,
                                   dtype=np.float64) + \
            self.grid.azi_offset+0.5
        normalized = unnormalized % 360
        np.testing.assert_equal(self.grid.azimuth, normalized)

    def test_coords_bounds_returns_two_arrays(self):
        azi_bounds, range_bounds = self.grid._coords_bounds()
        self.assertIsInstance(azi_bounds, np.ndarray)
        self.assertIsInstance(range_bounds, np.ndarray)

    def test_coords_bounds_calculate_azimuth_and_range(self):
        self.grid.nr_ranges = 50
        self.grid.nr_azi = 36
        self.grid.range_res = 2800
        azimuth = np.linspace(0, 360, self.grid.nr_azi+1, endpoint=True)
        azimuth = azimuth % 360
        range_dist = np.arange(self.grid.nr_ranges+1) * self.grid.range_res
        returned_coords = self.grid._coords_bounds()
        np.testing.assert_equal(azimuth, returned_coords[0])
        np.testing.assert_equal(range_dist, returned_coords[1])

    def test_center_distance_returns_centroid_range(self):
        grid_ranges = self.grid.coord_fields[1]
        np.testing.assert_equal(self.grid.center_distance, grid_ranges)
        np.testing.assert_equal(self.grid.center_distance[:, 0],
                                self.grid.center_distance[0, 0])

    def test_grid_as_multiindex_sets_product_as_values(self):
        right_multiindex = pd.MultiIndex.from_product(
            self.grid.coords, names=self.grid.coord_names
        )
        returned_multiindex = self.grid.get_multiindex()
        self.assertIsInstance(returned_multiindex, pd.MultiIndex)
        np.testing.assert_equal(
            returned_multiindex.values, right_multiindex.values
        )
        self.assertListEqual(returned_multiindex.names, right_multiindex.names)

    def test_get_azimuth_offset_returns_azi_offset_dataarray(self):
        self.assertEqual(self.grid.azi_offset,
                         self.grid.get_azimuth_offset().values)

    def test_get_azimuth_offset_sets_attributes(self):
        self.assertDictEqual(self.grid.get_azimuth_offset().attrs,
                             naming_convention[
                                 self.grid.get_azimuth_offset().name])

if __name__ == '__main__':
    unittest.main()
