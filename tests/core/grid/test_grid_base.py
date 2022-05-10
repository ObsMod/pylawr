#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os

# External modules
import numpy as np

# Internal modules
from pylawr.grid.base import BaseGrid
from pylawr.utilities.conventions import naming_convention

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestBaseGrid(unittest.TestCase):
    def setUp(self):
        self.grid = BaseGrid()
        self.test_pos = (53.1, 10, 43)

    def test_grids_equality(self):
        grid_temp = BaseGrid()
        self.assertTrue(self.grid == grid_temp)

    def test_grids_unequality(self):
        grid_temp = BaseGrid(self.test_pos)
        self.assertFalse(self.grid == grid_temp)

    def test_center_returns_private(self):
        self.assertEqual(self.grid.center, self.grid._center)
        self.grid._center = self.test_pos
        self.assertEqual(self.grid.center, self.grid._center)

    def test_center_sets_private(self):
        self.assertEqual(self.grid.center, self.grid._center)
        self.grid.center = self.test_pos
        self.assertEqual(self.grid._center, self.test_pos)

    def test_center_raises_typeerror_if_no_tuple(self):
        with self.assertRaises(TypeError):
            self.grid.center = 134

    def test_center_raises_valueerror_if_tuple_length_not_right(self):
        with self.assertRaises(ValueError):
            self.grid.center = (12, )
            self.grid.center = (214, 5356, 535, 75)

    def test_center_sets_default_height_if_not_given(self):
        self.grid.center = self.test_pos
        self.assertEqual(self.grid.center[2], self.test_pos[2])
        self.grid.center = self.test_pos[:2]
        self.assertEqual(len(self.grid.center), 3)
        self.assertEqual(self.grid.center[2], 0)

    def test_check_data_raises_typeerror_if_not_array(self):
        data_array = 12435
        with self.assertRaises(TypeError):
            self.grid._check_data(data_array)

    def test_check_data_raises_valueerror_if_wrong_shape(self):
        self.grid._data_shape = (5, 10)
        data = np.arange(50).reshape(self.grid._data_shape)
        self.grid._check_data(data)
        with self.assertRaises(ValueError):
            data = np.arange(20).reshape(2, 10)
            self.grid._check_data(data)

    def test_altitude_returns_altitude_if_set(self):
        altitude = 500
        self.grid._altitude = altitude
        self.assertEqual(altitude, self.grid.altitude)

    def test_altitude_calls_calc_altitude_if_altitude_is_none(self):
        def calc_altitude(*args):
            return 500
        self.grid._data_shape = 1
        self.grid._calc_altitude = calc_altitude
        self.grid._altitude = None
        self.assertEqual(self.grid.altitude, calc_altitude())

    def test_altitude_sets_constant_value_for_nonnone_shape(self):
        def calc_altitude(*args):
            return np.ones(self.grid.grid_shape) * 500

        self.grid._data_shape = (50, 50)
        self.grid._calc_altitude = calc_altitude
        self.grid._altitude = None
        self.assertTrue(np.array_equal(self.grid.altitude, calc_altitude()))

    def test_altitude_checks_shape(self):
        self.grid._data_shape = (500, 500)
        with self.assertRaises(ValueError):
            self.grid.altitude = np.ones((10, 10)) * 500

    def test_calc_altitude_for_one_distance(self):
        radar_distance = 10000.
        re = 6370040
        ke = 4. / 3.

        altitude = (
                np.sqrt(
                    radar_distance * radar_distance
                    + re * re * ke * ke
                    + 2 * re * ke *
                    radar_distance * np.sin(
                        np.deg2rad(self.grid.beam_elevation))
                ) - re * ke + self.grid.center[2]
        )

        np.testing.assert_almost_equal(
            self.grid._calc_altitude(radar_distance=radar_distance,
                                     radar_height=self.grid.center[2],
                                     earth_radius=re, ke=ke),
            altitude,
            decimal=3)

    def test_calc_altitude_for_distance_array(self):
        dist = np.linspace(0, 10000, 11)
        radar_distance = np.tile(dist, (10, 1))
        re = 6370040
        ke = 4. / 3.

        altitude = (
                np.sqrt(
                    radar_distance * radar_distance
                    + re * re * ke * ke
                    + 2 * re * ke *
                    radar_distance * np.sin(
                        np.deg2rad(self.grid.beam_elevation))
                ) - re * ke
        )

        np.testing.assert_array_almost_equal(
            self.grid._calc_altitude(radar_distance=radar_distance,
                                     earth_radius=re, ke=ke),
            altitude,
            decimal=3)

    def test_altitude_raises_typeerror_if_not_numpy_or_float(self):
        with self.assertRaises(TypeError):
            self.grid.altitude = 'bla'

    def test_coords_returns_coords_if_set(self):
        coords = 500
        self.grid._coords = coords
        self.assertEqual(coords, self.grid.coords)

    def test_coords_calls_calc_coords_if_coords_is_none(self):
        def calc_coords():
            return 500
        self.grid._calc_coordinates = calc_coords
        self.grid._coords = None
        self.assertEqual(self.grid.coords, calc_coords())

    def test_coord_names_return_private_coord_names(self):
        self.grid._coord_names = ('bla', 'blub')
        self.assertSequenceEqual(self.grid._coord_names, self.grid.coord_names)

    def test_beam_elevation_field_returns_repeated_beam_elevation_float(self):
        self.grid._data_shape = (10, 20)
        self.grid._beam_elevation = 10
        returned_field = self.grid.beam_elevation_field
        data_size = np.product(self.grid._data_shape)
        repeat_times = data_size/np.array(self.grid._beam_elevation).size
        repeated_f = np.array(self.grid._beam_elevation).repeat(repeat_times)
        repeated_f = repeated_f.reshape(self.grid._data_shape)
        np.testing.assert_equal(returned_field, repeated_f)

    def test_grid_shape_returns_grid_shape(self):
        self.grid._data_shape = (10, 20)
        self.assertSequenceEqual(self.grid.grid_shape, (10, 20))

    def test_grid_size_returns_product_of_shape(self):
        self.grid._data_shape = (10, 20)
        right_size = np.prod(self.grid._data_shape)
        self.assertEqual(self.grid.size, right_size)

    def test_beam_elevation_returns_private(self):
        beam_elevation = 10.
        self.grid._beam_elevation = beam_elevation
        self.assertAlmostEqual(self.grid.beam_elevation, beam_elevation)

    def test_beam_elevation_setter_sets_private(self):
        beam_elevation = 10.
        self.grid._beam_elevation = None
        self.grid.beam_elevation = beam_elevation
        self.assertAlmostEqual(self.grid._beam_elevation, beam_elevation)

    def test_beam_elevation_raises_typeerror_if_not_integer_or_float(self):
        with self.assertRaises(TypeError):
            self.grid.beam_elevation = np.zeros([5, 5])

    def test_beam_elevation_get_normalized_float(self):
        self.grid.beam_elevation = 130
        self.assertEqual(self.grid.beam_elevation, 50)
        self.grid.beam_elevation = 50
        self.assertEqual(self.grid.beam_elevation, 50)

    def test_beam_elevation_uses_init_value(self):
        test_beam_ele = 5.
        grid = BaseGrid(beam_ele=test_beam_ele)
        self.assertEqual(test_beam_ele, grid.beam_elevation)

    def test_get_beam_elevation_returns_beam_elevation_dataarray(self):
        self.assertEqual(self.grid.beam_elevation,
                         self.grid.get_beam_elevation().values)

    def test_get_beam_elevation_sets_attributes(self):
        self.assertDictEqual(self.grid.get_beam_elevation().attrs,
                             naming_convention[
                                 self.grid.get_beam_elevation().name])

    def test_get_center_returns_center(self):
        center_ds = self.grid.get_center()
        self.assertTupleEqual(self.grid.center,
                         (center_ds.lat_center.values.item(),
                          center_ds.lon_center.values.item(),
                          center_ds.zsl_center.values.item()))

    def test_get_center_sets_attrs(self):
        center_ds = self.grid.get_center()
        for var_name in list(center_ds.keys()):
            self.assertDictEqual(center_ds[var_name].attrs,
                                 naming_convention[var_name])

    def test_calc_earth_radius(self):
        radius_equator = 6378.137
        radius_pole = 6356.752
        lat = 50.
        self.grid.center = (lat, 10., 5.)
        lat_rad = np.deg2rad(self.grid.center[0])
        radius = np.sqrt(
            (np.power(radius_equator, 4) * np.power(np.cos(lat_rad), 2) +
             np.power(radius_pole, 4) * np.power(np.sin(lat_rad), 2)) /
            (np.power(radius_equator * np.cos(lat_rad), 2) +
             np.power(radius_pole * np.sin(lat_rad), 2))) * 1e3
        self.assertEqual(radius, self.grid.earth_radius)

    def test_earth_radius_private_following_lazy_property(self):
        self.assertIsNone(self.grid._earth_radius)
        earth_radius = self.grid.earth_radius
        self.assertEqual(self.grid._earth_radius, earth_radius)

    def test_earth_radius_private_not_overwrite(self):
        earth_radius = 6371e3
        self.grid._earth_radius = earth_radius
        self.assertEqual(earth_radius, self.grid.earth_radius)


if __name__ == '__main__':
    unittest.main()
