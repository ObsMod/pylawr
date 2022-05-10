#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os

# External modules
import numpy as np
import pandas as pd

# Internal modules
from pylawr.grid.unstructured import UnstructuredGrid


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
rnd = np.random.RandomState(42)


class TestUnstructuredGrid(unittest.TestCase):
    def setUp(self):
        size_sample = (100, 1)
        lat_sample = rnd.uniform(low=53.4, high=53.75, size=size_sample)
        lon_sample = rnd.uniform(low=9.7, high=10.2, size=size_sample)
        alt_sample = rnd.uniform(low=0., high=1000., size=size_sample)
        self.coords = np.concatenate([lat_sample, lon_sample, alt_sample],
                                     axis=1)
        self.grid = UnstructuredGrid(in_coords=self.coords)

    def test_data_shape_is_coord_len(self):
        self.assertTupleEqual(self.grid._data_shape, (100,))
        self.coords = rnd.normal(scale=10, size=(1000, 3))
        self.grid = UnstructuredGrid(in_coords=self.coords)
        self.assertTupleEqual(self.grid._data_shape, (1000,))

    def test_coord_names_is_grid_tuple(self):
        self.assertTupleEqual(self.grid._coord_names, ('grid_cell', ))

    def test_calc_coords_returns_in_coords(self):
        np.testing.assert_equal(
            self.grid._calc_coordinates()[0], self.grid.in_coords[:, :2]
        )

    def test_coords2latlon_identuity_function(self):
        returned_lat_lon = self.grid._coords2latlon(self.grid.in_coords)
        np.testing.assert_equal(
            self.grid.in_coords[:, 0],
            returned_lat_lon[0]
        )
        np.testing.assert_equal(
            self.grid.in_coords[:, 1],
            returned_lat_lon[1]
        )

    def test_calc_altitude_uses_third_coords_entry(self):
        np.testing.assert_equal(
            self.grid._calc_altitude(), self.grid.in_coords[:, 2]
        )

    def test_calc_altitude_calculates_altitude_if_no_third_entry(self):
        self.grid.in_coords = self.grid.in_coords[:, :2]
        self.assertTrue(
            np.any(np.not_equal(self.grid.altitude, self.coords[:,2]))
        )

    def test_altitude_shape_and_is_not_none(self):
        self.assertEqual(self.grid.altitude.shape, self.grid.grid_shape)

    def test_lat_lon_works(self):
        right_lat_lon = self.grid._coords2latlon(self.grid.in_coords)
        np.testing.assert_equal(
            right_lat_lon[0], self.grid.lat_lon[0]
        )
        np.testing.assert_equal(
            right_lat_lon[1], self.grid.lat_lon[1]
        )

    def test_calc_coord_fields_returns_identity(self):
        np.testing.assert_equal(
            self.grid.in_coords[:, :2],
            self.grid._calc_coord_fields(*self.grid._calc_coordinates())[0]
        )

    def test_get_coordinates_returns_multiindex(self):
        right_multiindex = pd.MultiIndex.from_arrays(
            self.grid.in_coords[:, :2].T, names=('grid_lat', 'grid_lon')
        )
        returned_multiindex = self.grid.get_coordinates()['grid_cell'][1]
        pd.testing.assert_index_equal(right_multiindex, returned_multiindex)

    def test_get_multiindex_returns_multindex(self):
        right_multiindex = pd.MultiIndex.from_arrays(
            self.grid.in_coords[:, :2].T, names=('grid_lat', 'grid_lon')
        )
        returned_multiindex = self.grid.get_multiindex()
        pd.testing.assert_index_equal(right_multiindex, returned_multiindex)

    def test_center_distance_haversine_distance_with_center(self):
        coords1 = self.grid.lat_lon
        center = self.grid.center

        right_distance = self.grid._haversine_distance(coords1, center[:2])

        np.testing.assert_equal(self.grid.center_distance, right_distance)

    def test_center_distance_is_for_central_grid_zero(self):
        center_coords = np.reshape(self.grid.center, (1,3))
        grid = UnstructuredGrid(in_coords=center_coords)
        self.assertEqual(grid.center_distance, 0)

    def test_haversine_formula_calcs_dist_between_different_points(self):
        coords1 = self.grid.lat_lon
        coords2 = tuple(i+1 for i in coords1)
        lat1, lon1 = coords1 = np.deg2rad(coords1)
        lat2, lon2 = coords2 = np.deg2rad(coords2)
        dlat = coords1[0] - coords2[0]
        dlon = coords1[1] - coords2[1]

        calculated_distance = self.grid._haversine_formula(
            coords1, coords2, dlat, dlon
        )

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
            dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        right_distance = self.grid.earth_radius * c

        np.testing.assert_equal(calculated_distance, right_distance)

    def test_haversine_calcs_dist_between_different_points(self):
        coords1 = self.grid.lat_lon
        coords2 = tuple(i+1 for i in coords1)

        calculated_distance = self.grid._haversine_distance(coords1, coords2)

        coords1 = np.deg2rad(coords1)
        coords2 = np.deg2rad(coords2)
        dlat = coords1[0] - coords2[0]
        dlon = coords1[1] - coords2[1]
        right_distance = self.grid._haversine_formula(
            coords1, coords2, dlat, dlon
        )

        np.testing.assert_equal(calculated_distance, right_distance)


if __name__ == '__main__':
    unittest.main()
