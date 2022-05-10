#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
from unittest.mock import patch, PropertyMock
import logging
import os
from copy import deepcopy

# External modules
import numpy as np

# Internal modules
from pylawr.grid.cartesian import CartesianGrid


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCartesianGrid(unittest.TestCase):
    def setUp(self):
        self.grid = CartesianGrid(resolution=(100, 100), start=(-1000, -200),)

    def test_nr_points_get_private(self):
        self.grid._data_shape = None
        self.assertIsNone(self.grid.nr_points)
        self.grid._data_shape = 5
        self.assertEqual(self.grid.nr_points, 5)

    def test_nr_points_sets_tuple(self):
        self.grid._data_shape = None
        self.assertIsNone(self.grid.nr_points)
        self.grid._data_shape = 5
        self.grid.nr_points = (100, 300)
        self.assertEqual(self.grid._data_shape, (100, 300))

    def test_nr_points_sets_int_to_tuple(self):
        self.grid.nr_points = 100
        self.assertEqual(self.grid._data_shape, (100, 100))

    def test_nr_points_raises_value_error_if_float(self):
        with self.assertRaises(TypeError):
            self.grid.nr_points = 100.1

    def test_nr_points_takes_first_two_entries(self):
        self.grid.nr_points = (100, 200, 300)
        self.assertEqual(self.grid._data_shape, (100, 200))

    def test_start_get_private(self):
        self.grid._start = None
        self.assertIsNone(self.grid._start)
        self.grid._start = 5
        self.assertEqual(self.grid.start, 5)

    def test_start_sets_tuple(self):
        self.grid._start = None
        self.assertIsNone(self.grid.start)
        self.grid._start = 5
        self.grid.start = (100, 300)
        self.assertEqual(self.grid._start, (100, 300))

    def test_start_sets_int_to_tuple(self):
        self.grid.start = 100
        self.assertEqual(self.grid._start, (100, 100))

    def test_start_takes_first_two_entries(self):
        self.grid.start = (100, 200, 300)
        self.assertEqual(self.grid.start, (100, 200))

    def test_calc_altitude_sets_altitude(self):
        ke = 4. / 3.
        altitude = (
                np.sqrt(
                    self.grid.center_distance * self.grid.center_distance
                    + ke * ke
                    * self.grid.earth_radius * self.grid.earth_radius +
                    2. * self.grid.center_distance
                    * ke * self.grid.earth_radius *
                    np.sin(np.deg2rad(self.grid.beam_elevation))
                ) - (ke * self.grid.earth_radius) + self.grid.center[2]
        )
        np.testing.assert_almost_equal(
            self.grid.altitude,
            altitude,
            decimal=3)

    def test_altitude_has_data_shape(self):
        self.grid._altitude = None
        self.assertEqual(self.grid.altitude.shape,
                         self.grid._data_shape)

    def test_calc_coordinates_returns_tuple(self):
        returned = self.grid._calc_coordinates()
        self.assertIsInstance(returned, tuple)
        self.assertEqual(len(returned), 2)

    def test_calc_coordinates_returns_y_and_x(self):
        y, x = self.grid._calc_coordinates()
        stepsy = self.grid.resolution[0]*self.grid.nr_points[0]
        right_y = np.arange(
            self.grid.start[0], self.grid.start[0]+stepsy,
            self.grid.resolution[0]
        )
        stepsx = self.grid.resolution[1] * self.grid.nr_points[1]
        right_x = np.arange(
            self.grid.start[1], self.grid.start[1]+stepsx,
            self.grid.resolution[1]
        )
        np.testing.assert_equal(y, right_y)
        np.testing.assert_equal(x, right_x)

    def test_calc_coords_bounds_returns_bounds(self):
        y, x = self.grid._coords_bounds()

        step_widthy = self.grid.resolution[0]*(self.grid.nr_points[0]+1)
        step_widthx = self.grid.resolution[1]*(self.grid.nr_points[1]+1)
        right_y = np.arange(-1050, -1050+step_widthy, 100)
        right_x = np.arange(-250, -250+step_widthx, 100)

        np.testing.assert_equal(y, right_y)
        np.testing.assert_equal(x, right_x)

    def test_meters2deg_returns_float(self):
        returned_m2deg = self.grid.meters2deg

        earth_peri = self.grid.earth_radius * 2 * np.pi
        right_m2deg = 360 / earth_peri

        self.assertIsInstance(returned_m2deg, float)
        self.assertEqual(right_m2deg, returned_m2deg)

    def test_meters2deg_uses_earth_radius(self):
        old_m2deg = deepcopy(self.grid.meters2deg)
        self.grid._earth_radius = 1
        new_m2deg = deepcopy(self.grid.meters2deg)
        self.assertNotEqual(old_m2deg, new_m2deg)

    def test_center_distance(self):
        right_distance = np.sqrt(
            self.grid.coord_fields[0] ** 2 + self.grid.coord_fields[1] ** 2
        )
        np.testing.assert_equal(self.grid.center_distance, right_distance)

    def test_calc_rotated_pole_based_on_center(self):
        pole = self.grid.north_pole
        self.assertEqual(pole[0], 90-self.grid.center[0])
        self.assertEqual(pole[1], self.grid.center[1]-180)

    def test_calc_rotated_pole_raises_valueerror_if_not_first_quad(self):
        self.grid.center = (-53.5, -10)
        error_msg = 'The given center coordinates are not within the first ' \
                    'quadrant, this is not supported at the moment!'

        with self.assertRaises(ValueError) as e:
            _ = self.grid.north_pole
        self.assertEqual(str(e.exception), error_msg)

    @patch('pylawr.grid.cartesian.CartesianGrid.north_pole',
           new_callable=PropertyMock)
    def test_coords2latlon_access_north_pole(self, pole_patch):
        pole_patch.return_value = (90, 90)
        _ = self.grid._coords2latlon(*self.grid.coord_fields)
        pole_patch.assert_called_once()
    #
    # @patch('cartopy.crs.RotatedPole.__init__', return_value=None)
    # @patch('cartopy.crs.PlateCarree.__init__', return_value=None)
    # def test_coords2latlon_init_plate_carre(self, pc_patch, rp_patch):
    #     _ = self.grid._coords2latlon(*self.grid.coord_fields)
    #     rp_patch.assert_called_once_with(pole_latitude=90, pole_longitude=0)
    #     pc_patch.assert_called_once_with()

    @patch('cartopy.crs.PlateCarree.transform_points',
           return_value=np.array([[1, 1]]))
    def test_coords2latlon_transform_points_called(self, pc_patch):
        _ = self.grid._coords2latlon(*self.grid.coord_fields)
        pc_patch.assert_called_once()

    @patch('pylawr.grid.cartesian.CartesianGrid.meters2deg',
           return_value=1, new_callable=PropertyMock)
    def test_coords2latlon_convert_coords_to_deg(self, m2deg_patch):
        _ = self.grid._coords2latlon(*self.grid.coord_fields)
        self.assertEqual(m2deg_patch.call_count, 2)

    @patch('pylawr.grid.cartesian.CartesianGrid.north_pole',
           return_value=(90, 180), new_callable=PropertyMock)
    def test_coords2latlon_right_coords(self, *args):
        y, x = self.grid.coord_fields

        orig_lat = y * self.grid.meters2deg
        orig_lon = x * self.grid.meters2deg
        lat, lon = self.grid._coords2latlon(y, x)

        np.testing.assert_almost_equal(lat, orig_lat)
        np.testing.assert_almost_equal(lon, orig_lon)

    @patch('pylawr.grid.cartesian.CartesianGrid.north_pole',
           return_value=(36.5, -170), new_callable=PropertyMock)
    def test_coords2latlon_hh_test(self, *args):
        lat, lon = self.grid._coords2latlon(np.array([0]), np.array([0]))
        self.assertAlmostEqual(lat[0], 53.5)
        self.assertAlmostEqual(lon[0], 10)

    def test_coords2latlon_center_test(self):
        lat, lon = self.grid._coords2latlon(np.array([0]), np.array([0]))
        self.assertAlmostEqual(lat[0], self.grid.center[0])
        self.assertAlmostEqual(lon[0], self.grid.center[1])


if __name__ == '__main__':
    unittest.main()
