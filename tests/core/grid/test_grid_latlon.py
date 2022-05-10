#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os

# External modules
import numpy as np

# Internal modules
from pylawr.grid.latlon import LatLonGrid


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestLatLonGrid(unittest.TestCase):
    def setUp(self):
        self.grid = LatLonGrid(resolution=(0.01, 0.01), start=(9.85, 53.5), )

    def test_grid_coordnames_are_lat_lon(self):
        orig_names = ('latitude', 'longitude')
        self.assertTupleEqual(orig_names, self.grid._coord_names)

    def test_coords2latlon_returns_coords(self):
        coordinates = self.grid.coord_fields
        returned_latlon = self.grid._coords2latlon(*coordinates)

        self.assertEqual(len(coordinates), len(returned_latlon))
        for i, orig in enumerate(coordinates):
            np.testing.assert_equal(returned_latlon[i], orig)

    def test_latlon_to_rad_converts_given_lat_lon_2_rad(self):
        coordinates = self.grid.coord_fields
        coords_rad = np.array(tuple(map(np.deg2rad, coordinates)))
        returned_coords = np.array(self.grid._latlon2rad(*coordinates))
        np.testing.assert_equal(coords_rad, returned_coords)

    def test_get_distance_calculates_distance_between_p2_and_p1(self):
        coords_1 = np.array(tuple(map(np.deg2rad, self.grid.coord_fields)))
        point_2 = (0.5, 0.1)

        arr_p2 = np.array(point_2)[..., np.newaxis, np.newaxis]
        right_distance = arr_p2 - coords_1
        calculated_distance = np.array(
            self.grid._get_distance(coords_1, point_2)
        )
        np.testing.assert_equal(calculated_distance, right_distance)

    def test_haversine_formula_calcs_dist_between_different_points(self):
        coords1 = self.grid.coord_fields
        coords2 = tuple(i+1 for i in coords1)
        lat1, lon1 = coords1 = self.grid._latlon2rad(*coords1)
        lat2, lon2 = coords2 = self.grid._latlon2rad(*coords2)
        dlat, dlon = self.grid._get_distance(coords1, coords2)

        calculated_distance = self.grid._haversine_formula(
            coords1, coords2, dlat, dlon
        )

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
            dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        right_distance = self.grid.earth_radius * c

        np.testing.assert_equal(calculated_distance, right_distance)

    def test_haversine_calcs_dist_between_different_points(self):
        coords1 = self.grid.coord_fields
        coords2 = tuple(i+1 for i in coords1)

        calculated_distance = self.grid._haversine_distance(coords1, coords2)

        coords1 = self.grid._latlon2rad(*coords1)
        coords2 = self.grid._latlon2rad(*coords2)
        dlat, dlon = self.grid._get_distance(coords1, coords2)
        right_distance = self.grid._haversine_formula(
            coords1, coords2, dlat, dlon
        )

        np.testing.assert_equal(calculated_distance, right_distance)

    def test_calc_altitude_sets_altitude(self):
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

    def test_center_distance_haversine_distance_with_center(self):
        coords1 = self.grid.coord_fields
        center = self.grid.center

        right_distance = self.grid._haversine_distance(coords1, center[:2])

        np.testing.assert_equal(self.grid.center_distance, right_distance)

    def test_coord_bounds_small_grid_resolution(self):
        grid_fine = LatLonGrid(resolution=(0.0009, 0.001515),
                                start=(53.387, 9.673), nr_points=401)
        grid_coarse = LatLonGrid(resolution=(0.1, 0.1),
                                 start=(53.387, 9.673), nr_points=401)

        self.assertEqual(grid_fine.lat_lon_bounds[0].shape,
                         grid_coarse.lat_lon_bounds[0].shape)


if __name__ == '__main__':
    unittest.main()
