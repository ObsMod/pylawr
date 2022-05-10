#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 24.05.18
#
# Created for pattern
#
#
#
#    Copyright (C) {2018}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# System modules
import os
import unittest
import logging
from unittest.mock import patch

# External modules
import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter

# Internal modules
from pylawr.field import tag_array
from pylawr.grid import PolarGrid
from pylawr.grid.unstructured import UnstructuredGrid
from pylawr.functions.grid import get_masked_grid
from pylawr.remap import NearestNeighbor
from pylawr.transform.spatial.interpolation import Interpolator
from pylawr.utilities.helpers import polar_padding, create_array


BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


random_state = np.random.RandomState(42)

logging.basicConfig(level=logging.DEBUG)


class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self.filter = Interpolator(polar=False)
        # self.field = xr.open_dataarray(
        #     os.path.join(DATA_PATH, 'lawr_data_170914105829.nc')
        # )

        # self.field = self.field.lawr.set_variable('dbz')
        self.grid = PolarGrid()
        self.field = create_array(self.grid, const_val=20.)
        self.field = self.field.lawr.set_grid_coordinates(self.grid)

    def test_algorithm_property_returns_private(self):
        self.filter._algorithm = 123
        self.assertEqual(self.filter._algorithm, self.filter.algorithm)

    def test_algorithm_property_sets_private(self):
        algorithm = NearestNeighbor()
        self.assertNotEqual(id(self.filter._algorithm), id(algorithm))
        self.filter.algorithm = algorithm
        self.assertEqual(id(self.filter._algorithm), id(algorithm))

    def test_algorithm_checks_if_algorithm_or_none(self):
        err_msg = 'The given algorithm is not None or a valid remapping ' \
                  'algorithm'
        with self.assertRaises(TypeError) as e:
            self.filter.algorithm = 123
        self.assertEqual(str(e.exception), err_msg)

    def test_algorithm_sets_to_nearest_neighbor_with_one_neighbor_if_none(self):
        self.filter._algorithm = None
        self.assertIsNone(self.filter.algorithm)
        self.filter.algorithm = None
        self.assertIsInstance(self.filter.algorithm, NearestNeighbor)
        self.assertEqual(self.filter.algorithm.n_neighbors, 1)
        self.assertFalse(self.filter.algorithm.fitted)

    def test_get_predictor_mask_returns_good_values_as_true(self):
        self.field[:] = 32
        predictor_mask = self.filter._get_source_mask(self.field)
        self.assertTrue(np.all(predictor_mask))

    def test_get_predictor_mask_masks_nan_values(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))
        self.field = self.field.where(mask_array, drop=False)
        predictor_mask = self.filter._get_source_mask(self.field)
        self.assertIsInstance(predictor_mask, np.ndarray)
        self.assertEqual(predictor_mask.dtype, bool)
        self.assertFalse(np.any(predictor_mask[~mask_array]))

    @staticmethod
    def _create_refl_array():
        data = np.ones((1, 3, 3)) * (-32.5)
        field = xr.DataArray(data, dims=['time', 'azimuth', 'range'])
        return field

    def test_get_predictor_mask_interpolates_embedded(self):
        field = self._create_refl_array()
        field[..., 1, 1] = 30
        field[..., 0, 1] = 30
        field[..., :, 2] = 30
        field[..., :, 0] = np.nan
        predictor_mask = self.filter._get_source_mask(field)
        self.assertTrue(predictor_mask[..., 1, 1])

    def test_get_predictor_mask__not_interpolates_front(self):
        field = self._create_refl_array()
        field[..., 1, 1] = 30
        field[..., :, 2] = 30
        field[..., :, 0] = np.nan
        predictor_mask = self.filter._get_source_mask(field)
        self.assertFalse(predictor_mask[..., 1, 1])

    def test_get_predictor_mask_not_interpolate_corner(self):
        field = self._create_refl_array()
        field[..., 1, 1] = 30
        field[..., :1, 2] = 30
        field[..., :, 0] = np.nan

        predictor_mask = self.filter._get_source_mask(field)
        self.assertFalse(predictor_mask[..., 1, 1])

    def test_get_predictor_mask_not_interpolate_front_with_no_rain(self):
        field = self._create_refl_array()
        field[..., 1, 1] = 30
        field[..., :, 2] = 0
        field[..., :, 0] = np.nan

        predictor_mask = self.filter._get_source_mask(field)
        self.assertFalse(predictor_mask[..., 1, 1])

    def test_get_predictor_mask_interpolate_if_not_nan(self):
        field = self._create_refl_array()
        field[..., 1, 1] = 30
        field[..., :, 2] = 0
        predictor_mask = self.filter._get_source_mask(field)
        self.assertTrue(predictor_mask[..., 1, 1])

    def test_get_predictor_mask_uses_polar_padding_if_polar(self):
        field_padded_data = polar_padding(self.field.values, pad_size=(1, 1))
        field_padded = xr.DataArray(
            field_padded_data,
            dims=self.field.dims
        )
        mask_padded = self.filter._get_source_mask(field_padded)

        self.filter.polar = True
        mask_polar = self.filter._get_source_mask(self.field)
        np.testing.assert_equal(mask_polar, mask_padded[..., 1:-1, 1:-1])

    def test_get_predictor_mask_sets_reflectivity_under_5_to_predictor(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))

        self.field[:] = -20
        self.field = self.field.where(mask_array, drop=False)
        predictor_mask = self.filter._get_source_mask(self.field)
        np.testing.assert_equal(predictor_mask, mask_array)

    def test_get_target_mask_returns_mask_for_nan_values(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))
        self.field = self.field.where(mask_array, drop=False)
        target_mask = self.filter._get_target_mask(self.field)
        np.testing.assert_equal(target_mask, ~mask_array)

    def test_get_masked_grid_returns_unstructured_grid(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.5, 0.5))
        mask_array = mask_array.squeeze()
        masked_grid = get_masked_grid(self.grid, mask_array)
        masked_vals = np.array(self.grid.lat_lon)[..., mask_array]
        np.testing.assert_equal(
            np.array(masked_grid.coords[0]), masked_vals.T
        )
        self.assertIsInstance(masked_grid, UnstructuredGrid)

    def test_get_masked_grid_returns_unstrucutred_altitude(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.5, 0.5))
        mask_array = mask_array.squeeze()
        masked_grid = get_masked_grid(self.grid, mask_array)
        masked_vals = self.grid.altitude[..., mask_array]
        np.testing.assert_equal(
            masked_grid.altitude, masked_vals
        )

    def test_get_masked_grid_sets_origin_center(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.5, 0.5))
        mask_array = mask_array.squeeze()
        masked_grid = get_masked_grid(self.grid, mask_array)
        self.assertTupleEqual(masked_grid.center, self.grid.center)

    def test_transform_uses_get_grid(self):
        with patch('pylawr.transform.spatial.interpolation.get_verified_grid',
                   return_value=self.grid) as p:
            _ = self.filter.transform(self.field, grid=self.grid)
            p.assert_called_once()

    def test_transform_calls_source_mask(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))

        self.field[:] = 32
        self.field = self.field.where(mask_array, drop=False)
        predictor_mask = self.filter._get_source_mask(self.field)
        with patch('pylawr.transform.spatial.interpolation.Interpolator.'
                   '_get_source_mask',
                   return_value=predictor_mask) as p:
            _ = self.filter.transform(self.field, grid=self.grid)
            p.assert_called_once()

    def test_transform_calls_target_mask(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))

        self.field[:] = 32
        self.field = self.field.where(mask_array, drop=False)
        target_mask = self.filter._get_target_mask(self.field)
        with patch('pylawr.transform.spatial.interpolation.Interpolator.'
                   '_get_target_mask',
                   return_value=target_mask) as p:
            _ = self.filter.transform(self.field, grid=self.grid)
            p.assert_called()
            self.assertEqual(p.call_count, 2)

    def test_transform_calls_fill_zeros(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))

        self.field[:] = 32
        self.field = self.field.where(mask_array, drop=False)
        with patch('pylawr.transform.spatial.interpolation.Interpolator.'
                   '_prefill_noninterp_vals',
                   return_value=self.field) as p:
            _ = self.filter.transform(self.field, grid=self.grid)
            p.assert_called()

    def test_transform_calls_get_source_grid(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))

        self.field[:] = 32
        self.field = self.field.where(mask_array, drop=False)
        predictor_mask = self.filter._get_source_mask(self.field).squeeze()
        target_mask = self.filter._get_target_mask(self.field).squeeze()
        predictor_grid = get_masked_grid(self.grid, predictor_mask)
        target_grid = get_masked_grid(self.grid, target_mask)
        target = 'pylawr.transform.spatial.interpolation.get_masked_grid'
        with patch(target, side_effect=[predictor_grid, target_grid]) as p:
            _ = self.filter.transform(self.field, grid=self.grid)
            p.assert_called()
            self.assertGreaterEqual(p. call_count, 1)
            np.testing.assert_equal(p.call_args_list[0][1]['mask_array'],
                                    predictor_mask)
            self.assertEqual(id(p.call_args_list[0][1]['origin_grid']),
                             id(self.grid))

    def test_valid_mask_returns_zero_mask(self):
        self.filter = Interpolator(zero_thres=0.5, polar=False)
        self.field[:] = 9999999
        zero_mask = random_state.choice(np.array([False, True], dtype=bool),
                                        size=self.field.shape, p=(0.5, 0.5))
        self.field = self.field.where(zero_mask, -32.5, drop=False)
        rain_mask = (self.field > 5).astype(float)
        np.testing.assert_equal(rain_mask, zero_mask)
        valid_mean = uniform_filter(
            zero_mask.astype(float), size=(1, 11, 11),
            mode='reflect'
        )
        right_mask = valid_mean > self.filter.zero_thres
        returned_mask = self.filter._get_valid_mask(self.field)
        np.testing.assert_equal(right_mask.squeeze(), returned_mask)

    def test_valid_mask_uses_polar_padding(self):
        self.filter = Interpolator(zero_thres=0.5)
        self.field[:] = 9999999
        zero_mask = random_state.choice(np.array([False, True], dtype=bool),
                                        size=self.field.shape, p=(0.5, 0.5))
        self.field = self.field.where(zero_mask, -32.5, drop=False)
        rain_mask = (self.field > 5).astype(float)
        np.testing.assert_equal(rain_mask, zero_mask)
        valid_mean = polar_padding(zero_mask.astype(float), pad_size=(5, 5))
        valid_mean = uniform_filter(
            valid_mean, size=(1, 11, 11),
            mode='reflect'
        )[..., 5:-5, 5:-5]
        right_mask = valid_mean > self.filter.zero_thres
        returned_mask = self.filter._get_valid_mask(self.field)
        np.testing.assert_equal(right_mask.squeeze(), returned_mask)

    def test_zero_fill_values_fills_where_zero_and_target(self):
        self.field[:] = 9999999
        zero_mask = random_state.choice(np.array([False, True], dtype=bool),
                                        size=self.field.shape, p=(0.7, 0.3))
        self.field = self.field.where(zero_mask, -32.5, drop=False)
        array_mask = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.3, 0.7))
        self.field = self.field.where(array_mask, drop=False)
        returned_field = self.filter._prefill_noninterp_vals(self.field)

        valid_mask = self.filter._get_valid_mask(self.field)
        zero_mask = np.logical_and(~array_mask, ~valid_mask)
        self.field.values[zero_mask] = -32.5
        xr.testing.assert_equal(returned_field, self.field)

    def test_transform_calls_get_target_grid(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))

        self.field[:] = 32
        self.field = self.field.where(mask_array, drop=False)
        predictor_mask = self.filter._get_source_mask(self.field).squeeze()
        target_mask = self.filter._get_target_mask(self.field).squeeze()
        predictor_grid = get_masked_grid(self.grid, predictor_mask)
        target_grid = get_masked_grid(self.grid, target_mask)
        target = 'pylawr.transform.spatial.interpolation.get_masked_grid'
        with patch(target, side_effect=[predictor_grid, target_grid]) as p:
            _ = self.filter.transform(self.field, grid=self.grid)
            p.assert_called()
            self.assertEqual(p. call_count, 2)
            np.testing.assert_equal(p.call_args_list[1][1]['mask_array'],
                                    target_mask)
            self.assertEqual(id(p.call_args_list[1][1]['origin_grid']),
                             id(self.grid))

    def test_transform_calls_fit_from_remapper(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))

        self.field[:] = 32
        self.field = self.field.where(mask_array, drop=False)
        predictor_mask = self.filter._get_source_mask(self.field).squeeze()
        target_mask = self.filter._get_target_mask(self.field).squeeze()
        predictor_grid = get_masked_grid(self.grid, predictor_mask)
        target_grid = get_masked_grid(self.grid, target_mask)
        _ = self.filter.transform(self.field, grid=self.grid)
        self.assertTrue(self.filter.algorithm.fitted)
        np.testing.assert_equal(
            np.concatenate(self.filter.algorithm._grid_in.coords),
            np.concatenate(predictor_grid.coords),
        )
        np.testing.assert_equal(
            np.concatenate(self.filter.algorithm._grid_out.coords),
            np.concatenate(target_grid.coords),
        )

    def test_transform_interpolates_data(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))
        self.field[:] = 32
        self.field = self.field.where(mask_array, drop=False)
        filled_values = self.filter.transform(self.field, grid=self.grid)
        target_mask = self.filter._get_target_mask(self.field).squeeze()
        nan_vals = np.sum(np.isnan(filled_values.values[..., target_mask]))
        self.assertEqual(nan_vals, 0)
        np.testing.assert_equal(filled_values.values, 32)

    def test_interpolate_transforms_array_to_dbz(self):
        mask_array = random_state.choice(np.array([False, True], dtype=bool),
                                         size=self.field.shape, p=(0.1, 0.9))
        self.field[:] = 32
        dbz_field = self.field.copy()
        self.field = self.field.lawr.to_z()
        self.field = self.field.where(mask_array, drop=False)
        with patch('pylawr.field.RadarField.to_dbz',
                   return_value=dbz_field) as p:
            _ = self.filter.transform(self.field, grid=self.grid)
        p.assert_called_with()

    def test_transform_sets_nan_to_zero(self):
        self.field[:] = np.nan
        filled_values = self.filter.transform(self.field, grid=self.grid)
        np.testing.assert_array_equal(filled_values, -32.5)

    def test_transform_keeps_grid(self):
        transformed_array = self.filter.transform(self.field, grid=self.grid)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.field.lawr.grid)

    def test_transform_keeps_attributes(self):
        tag_array(self.field, 'test-tag')

        transformed_array = self.filter.transform(self.field, grid=self.grid)

        self.assertIn('test-tag', transformed_array.attrs["tags"])

    def test_transform_uses_covariance_if_available(self):
        self.field[:] = 10
        self.field[..., :10] = np.nan
        covariance = np.ones((1, 360, 10))
        covariance[..., :5] = 30
        covariance = covariance.flatten()
        self.filter.zero_thres = -9999
        self.filter.algorithm.covariance = covariance
        transformed_array = self.filter.transform(self.field, grid=self.grid)
        np.testing.assert_equal(transformed_array.values[..., :5], -32.5)
        np.testing.assert_equal(transformed_array.values[..., 5:], 10)


if __name__ == '__main__':
    unittest.main()
