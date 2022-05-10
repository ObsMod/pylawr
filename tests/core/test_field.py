#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import os
from random import randint
import unittest
from unittest.mock import patch
import warnings
from copy import deepcopy

# External modules
import numpy as np
import xarray as xr

# Internal modules
from pylawr.grid.polar import PolarGrid
from pylawr.field import RadarField, TAGS_KEY, TAGS_SEP
from pylawr.utilities.conventions import naming_convention
from pylawr.utilities.trafo import from_decibel_to_linear
from pylawr.utilities.helpers import create_array

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class FieldTestClass(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()
        self.array = create_array(grid=self.grid)


class FilterClass(object):
    def transform(self, data, **kwargs):
        return data


class FilterTwoClass(object):
    def __init__(self):
        self.grid = PolarGrid(range_res=100)

    def transform(self, data, **kwargs):
        return data, self.grid


class TestRadarField(FieldTestClass):
    def test_array_has_accessor(self):
        self.assertTrue(hasattr(self.array, 'lawr'))

    def test_accessor_has_array_as_data(self):
        self.assertEqual(id(self.array.lawr._data), id(self.array))

    def test_data_returns_private(self):
        self.assertEqual(id(self.array.lawr.data), id(self.array.lawr._data))
        self.array.lawr._data = None
        self.assertIsNone(self.array.lawr.data)

    def test_grid_returns_grid(self):
        self.array.lawr._grid = self.grid
        self.assertEqual(self.array.lawr.grid, self.grid)

    def test_grid_could_be_set(self):
        self.array.lawr._grid = None
        self.assertIsNone(self.array.lawr._grid)
        self.array.lawr.grid = self.grid
        self.assertEqual(self.array.lawr._grid, self.grid)

    def test_grid_raises_typeeerror_if_no_grid_is_set(self):
        self.array.lawr._grid = None
        with self.assertRaises(TypeError):
            _ = self.array.lawr.grid

    def test_grid_raises_typeerror_if_not_valid(self):
        self.array.lawr.grid = None
        with self.assertRaises(TypeError):
            self.array.lawr.grid = [4761357, ]

    def test_grid_raises_valueerror_if_shapes_are_not_equal(self):
        self.array.lawr.grid = self.grid
        self.grid._data_shape = (2, 3)
        with self.assertRaises(ValueError):
            self.array.lawr.grid = self.grid

    @patch('pylawr.field.RadarField.check_grid')
    def test_grid_calls_check_grid(self, checker_mock):
        self.array.lawr.set_grid_coordinates(self.grid)
        checker_mock.assert_called_with(self.grid)

    def test_set_grid_coordinates_cooordinates_returns_grid_array(self):
        returned_array = self.array.lawr.set_grid_coordinates(self.grid)
        self.assertEqual(id(returned_array.lawr.grid), id(self.grid))

    def test_set_grid_coordinates_coordinates_sets_names(self):
        self.array = self.array.rename({'azimuth': 'a', 'range': 'b'})
        old_dim_names = np.array(self.array.dims)
        true_dim_names = list(old_dim_names[:-2]) + list(self.grid.coord_names)
        np.testing.assert_equal(np.array(self.array.dims), old_dim_names)
        gridded_array = self.array.lawr.set_grid_coordinates(self.grid)
        np.testing.assert_equal(np.array(gridded_array.dims), true_dim_names)

    def test_set_grid_coordinates_sets_values(self):
        self.array['azimuth'] = np.arange(self.array['azimuth'].size)+1
        self.array['range'] = np.arange(self.array['range'].size)
        self.assertFalse(any([
            np.all(np.equal(self.array[dim].values, self.grid.coords[num]))
            for num, dim in enumerate(self.array.dims[-2:])])
        )
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.assertTrue(all([
            np.all(np.equal(self.array[dim].values, self.grid.coords[num]))
            for num, dim in enumerate(self.array.dims[-2:])])
        )

    def test_grid_to_array_returns_dataset(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        returned_ds = self.array.lawr.grid_to_array()
        self.assertIsInstance(returned_ds, xr.Dataset)

    def test_grid_to_array_adds_lat_lon_and_altitude(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        returned_ds = self.array.lawr.grid_to_array()
        merged_ds = xr.merge([
            self.array.to_dataset(name='dbz'),
            self.grid.get_lat_lon(),
            self.grid.get_altitude().to_dataset(name='altitude')
        ])
        xr.testing.assert_equal(merged_ds, returned_ds)

    def test_grid_to_array_adds_metadata_to_attrs(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        returned_ds = self.array.lawr.grid_to_array()
        self.assertSequenceEqual(returned_ds.attrs['grid_center'],
                                 self.grid.center)
        self.assertEqual(returned_ds.attrs['grid_type'],
                         self.grid.__class__.__name__)

    def test_filter_returns_dataarray(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        returned_value = self.array.lawr.filter(FilterClass())
        self.assertIsInstance(returned_value, xr.DataArray)

    def test_filter_adds_grid_to_dataarray(self):
        self.array.lawr._grid = None
        self.array['azimuth'] = np.arange(self.array['azimuth'].size)+1
        self.array['range'] = np.arange(self.array['range'].size)
        self.assertIsNone(self.array.lawr._grid)
        self.array.lawr._grid = self.grid
        returned_array = self.array.lawr.filter(FilterClass())
        self.assertEqual(id(self.grid), id(returned_array.lawr.grid))

    def test_filter_sets_returned_grid(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        new_filter = FilterTwoClass()
        returned_array = self.array.lawr.filter(new_filter)
        self.assertEqual(id(new_filter.grid), id(returned_array.lawr._grid))
        gridded_array = self.array.lawr.set_grid_coordinates(new_filter.grid)
        xr.testing.assert_identical(returned_array, gridded_array)

    def test_set_variable_checks_variable_name(self):
        wrong_name = 'refl3215456'
        with self.assertRaises(KeyError) as e:
            self.array.lawr.set_variable(wrong_name)
            self.assertEqual(
                e.msg,
                'The given variable name: {0:s} cannot be found within '
                'the naming convention.\nAvailable variables: {1:s}'.format(
                    wrong_name, ','.join(naming_convention.keys())
                )
            )

    def test_set_variable_sets_attrs(self):
        name = 'z'
        out_field = self.array.lawr.set_variable(name)
        attrs = {a: out_field.attrs[a] for a in naming_convention[name]}
        self.assertDictEqual(attrs, naming_convention[name])

    def test_set_variable_sets_grid(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        out_field = self.array.lawr.set_variable('z')
        self.assertEqual(id(out_field.lawr.grid), id(self.array.lawr.grid))

    def test_set_variable_not_raises_error_if_no_grid(self):
        _ = self.array.lawr.set_variable('z')

    def test_set_variable_sets_name_to_given_name(self):
        out_field = self.array.lawr.set_variable('z')
        self.assertEqual(out_field.name, 'z')


class TestRadarFieldTags(unittest.TestCase):
    def setUp(self):
        random = np.random.random((5,5))
        array = xr.DataArray(random)
        self.f = RadarField(array)
        self.tags = list("WHATTALOTTALETTERS")
        self.tag = "tag"

    def tearDown(self):
        self.assertNoEmptyTags()

    def assertNoEmptyTags(self):
        self.assertNotIn("", self.f.tags)

    def test_tags_getter_sets_empty_tag_attr_if_not_there_yet(self):
        self.assertFalse(TAGS_KEY in self.f.data.attrs)
        self.f.tags
        self.assertTrue(TAGS_KEY in self.f.data.attrs)
        self.assertEqual(self.f.data.attrs[TAGS_KEY], '')

    def test_tags_getter_returns_split_list(self):
        self.f.data.attrs[TAGS_KEY] = TAGS_SEP.join(self.tags)
        self.assertEqual(self.f.tags, self.tags)

    def test_tags_setter_sets_attr(self):
        self.f.tags = self.tags
        self.assertEqual(self.f.data.attrs[TAGS_KEY], TAGS_SEP.join(self.tags))

    def test_add_tag_adds_tag(self):
        self.f.add_tag(self.tag)
        self.assertIn(self.tag, self.f.tags)

    def test_remove_tag_removes_tag_incl_duplicates(self):
        self.f.tags = self.tags
        tags = self.f.tags.copy()
        tag = tags.pop(randint(0, len(tags)-1))
        tags = [x for x in tags if x != tag]
        self.f.remove_tag(tag)
        self.assertListEqual(self.f.tags, tags)


class TestRadarFieldUnit(FieldTestClass):
    def setUp(self):
        super().setUp()
        self.zr_convertible = ['rr', 'z']

    def test_zr_convert_z_to_r(self):
        self.array = self.array.lawr.set_variable('z')
        r_field = (self.array / 200.) ** (1. / 1.6)
        out_field = self.array.lawr.zr_convert(a=200., b=1.6)
        xr.testing.assert_equal(out_field, r_field)

    def test_zr_sets_attrs_from_original(self):
        self.array = self.array.lawr.set_variable('z')
        out_field = self.array.lawr.zr_convert()
        out_attrs = {a: out_field.attrs[a] for a in out_field.attrs
                     if a not in ['units', 'standard_name', 'short_name']}
        orig_attrs = {a: self.array.attrs[a] for a in self.array.attrs
                      if a not in ['units', 'standard_name', 'short_name']}
        self.assertDictEqual(out_attrs, orig_attrs)

    def test_zr_uses_a_b(self):
        self.array = self.array.lawr.set_variable('z')
        a = 12
        b = 2
        out_field = self.array.lawr.zr_convert(a=a, b=b)
        r_field = (self.array / a) ** (1. / b)
        xr.testing.assert_equal(out_field, r_field)

    def test_zr_sets_grid(self):
        self.array = self.array.lawr.set_variable('z')
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        out_field = self.array.lawr.zr_convert()
        self.assertEqual(id(out_field.lawr.grid), id(self.array.lawr.grid))

    def test_zr_not_raises_error_if_no_grid(self):
        self.array = self.array.lawr.set_variable('z')
        self.array = self.array.lawr.zr_convert()

    def test_zr_sets_unit_to_r(self):
        self.array = self.array.lawr.set_variable('z')
        out_field = self.array.lawr.zr_convert()
        self.assertEqual(out_field.attrs['units'], 'mm/h')

    def test_zr_sets_attrs_to_rate(self):
        self.array = self.array.lawr.set_variable('z')
        out_field = self.array.lawr.zr_convert()
        attrs = {a: out_field.attrs[a] for a in naming_convention['rr']}
        self.assertDictEqual(attrs, naming_convention['rr'])

    def test_zr_inverse_from_r_to_z(self):
        self.array = self.array.lawr.set_variable('rr')
        out_field = self.array.lawr.zr_convert(inverse=True, a=200, b=1.6)
        z_field = 200 * self.array ** 1.6
        xr.testing.assert_equal(out_field, z_field)

    def test_zr_inverse_uses_a_b(self):
        self.array = self.array.lawr.set_variable('rr')
        a = 12
        b = 2
        out_field = self.array.lawr.zr_convert(a=a, b=b, inverse=True)
        r_field = a * self.array ** b
        xr.testing.assert_equal(out_field, r_field)

    def test_zr_inverse_sets_right_var(self):
        name = 'z'
        self.array = self.array.lawr.set_variable('rr')
        out_field = self.array.lawr.zr_convert(inverse=True)
        attrs = {
            a: out_field.attrs[a] for a in naming_convention[name]
        }
        self.assertDictEqual(attrs, naming_convention[name])
        self.assertEqual(out_field.name, name)

    def test_zr_inverse_sets_grid(self):
        self.array = self.array.lawr.set_variable('rr')
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        out_field = self.array.lawr.zr_convert(inverse=True)
        self.assertEqual(id(out_field.lawr.grid), id(self.array.lawr.grid))

    def test_zr_raises_userwarning_if_not_z_or_r(self):
        not_convertible = [v for v in naming_convention
                           if v not in self.zr_convertible]
        for var in not_convertible:
            self.array = self.array.lawr.set_variable(var)
            with warnings.catch_warnings(record=True) as w:
                warn_msg = '{0:s} cannot converted to {1:s}, I will return ' \
                           'the input array!'.format(var, 'rr')
                returned_array = self.array.lawr.zr_convert()
                self.assertTrue(issubclass(w[-1].category, UserWarning))
                self.assertEqual(len(w), 1)
                self.assertEqual(str(w[-1].message), warn_msg)
                xr.testing.assert_identical(returned_array, self.array)

    def test_zr_raises_userwarning_if_already_the_var(self):
        self.array = self.array.lawr.set_variable('rr')
        with warnings.catch_warnings(record=True) as w:
            warn_msg = 'The variable is already {0:s}, I will return the ' \
                       'input array!'.format('rr')
            returned_array = self.array.lawr.zr_convert()
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[-1].message), warn_msg)
            xr.testing.assert_identical(returned_array, self.array)

    def test_zr_inverse_raises_userwarning_if_already_the_var(self):
        self.array = self.array.lawr.set_variable('z')
        with warnings.catch_warnings(record=True) as w:
            warn_msg = 'The variable is already {0:s}, I will return the ' \
                       'input array!'.format('z')
            returned_array = self.array.lawr.zr_convert(inverse=True)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[-1].message), warn_msg)
            xr.testing.assert_identical(returned_array, self.array)

    def test_convert_db_to_linear(self):
        out_array = self.array.lawr.db_to_linear()
        right_array = 10. ** (self.array / 10.)
        xr.testing.assert_equal(out_array, right_array)

    def test_convert_db_inverse_linear_to_db(self):
        self.array = self.array.lawr.set_variable('z')
        out_array = self.array.lawr.db_to_linear(inverse=True)

        right_array = np.clip(
            10. * np.log10(
                np.clip(self.array, 0, np.inf)),
            -32.5, np.inf
        )

        xr.testing.assert_equal(out_array, right_array)

    def test_convert_db_sets_orig_attr(self):
        out_field = self.array.lawr.db_to_linear()
        out_attrs = {a: out_field.attrs[a] for a in out_field.attrs
                     if a not in ['units', 'standard_name', 'short_name']}
        orig_attrs = {a: self.array.attrs[a] for a in self.array.attrs
                      if a not in ['units', 'standard_name', 'short_name']}
        self.assertDictEqual(out_attrs, orig_attrs)

    def test_convert_db_sets_right_var(self):
        out_field = self.array.lawr.db_to_linear()
        attrs = {a: out_field.attrs[a] for a in
                 naming_convention['z']}
        self.assertDictEqual(attrs, naming_convention['z'])

    def test_convert_db_sets_grid(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        out_field = self.array.lawr.db_to_linear()
        self.assertEqual(id(out_field.lawr.grid), id(self.array.lawr.grid))

    def test_convert_db_not_raises_error_if_no_grid(self):
        self.array = self.array.lawr.db_to_linear()

    def test_convert_db_inverse_sets_right_var(self):
        self.array = self.array.lawr.set_variable('z')
        out_field = self.array.lawr.db_to_linear(inverse=True)
        attrs = {a: out_field.attrs[a] for a in
                 naming_convention['dbz']}
        self.assertDictEqual(attrs, naming_convention['dbz'])

    def test_convert_db_warning_if_not_available(self):
        self.array.name = 'Test'
        with warnings.catch_warnings(record=True) as w:
            warn_msg = 'The variable `{0:s}` cannot be converted, I will ' \
                       'return the input array!'.format(self.array.name)
            returned_array = self.array.lawr.db_to_linear()
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[-1].message), warn_msg)
            xr.testing.assert_identical(returned_array, self.array)

    def test_convert_db_inverse_warning_if_not_available(self):
        self.array.name = 'Test'
        with warnings.catch_warnings(record=True) as w:
            warn_msg = 'The variable `{0:s}` cannot be converted, I will ' \
                       'return the input array!'.format(self.array.name)
            returned_array = self.array.lawr.db_to_linear(inverse=True)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[-1].message), warn_msg)
            xr.testing.assert_identical(returned_array, self.array)

    def test_convert_db_none_returns_linear_refl(self):
        self.array.name = None
        with warnings.catch_warnings(record=True) as w:
            returned_array = self.array.lawr.db_to_linear()
            self.assertFalse(w)
        self.array = self.array.lawr.set_variable('dbz')
        right_array = self.array.lawr.db_to_linear()
        xr.testing.assert_identical(returned_array, right_array)

    def test_convert_field_apply_func_on_field(self):
        func_a = lambda x: x**2

        target = func_a(self.array)
        out_array = self.array.lawr._convert_field(func_a)

        xr.testing.assert_equal(target, out_array)

    def test_convert_field_sets_target_var(self):
        func_a = lambda x: x**2
        target_var = 'rr'
        convention = naming_convention[target_var]

        right_array = func_a(self.array)
        right_array = right_array.lawr.set_variable(target_var)

        out_array = self.array.lawr._convert_field(func_a, target_var)
        out_attrs = {a: out_array.attrs[a] for a in convention}

        self.assertEqual(right_array.name, out_array.name)
        self.assertDictEqual(convention, out_attrs)

    def test_convert_field_sets_old_attrs(self):
        self.array.attrs['test'] = 'bla'
        func_a = lambda x: x**2
        target_var = 'rr'

        right_array = func_a(self.array)
        right_array.attrs = self.array.attrs
        right_array = right_array.lawr.set_variable(target_var)

        out_array = self.array.lawr._convert_field(func_a, target_var)

        self.assertDictEqual(right_array.attrs, out_array.attrs)

    def test_convert_field_sets_grid_if_set(self):
        func_a = lambda x: x**2
        target_var = 'rr'
        self.array = self.array.lawr.set_grid_coordinates(self.grid)

        out_array = self.array.lawr._convert_field(func_a, target_var)

        self.assertEqual(id(self.grid), id(out_array.lawr.grid))

    def test_to_z_converts_trafo_var_to_z_attrs_check(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        trafo_vars = list(from_decibel_to_linear.keys()) + \
                     list(from_decibel_to_linear.values())
        for var in trafo_vars:
            tmp_array = self.array.lawr.set_variable(var)
            out_array = tmp_array.lawr.to_z()

            right_attrs = tmp_array.lawr.set_variable('z')
            self.assertDictEqual(right_attrs.attrs, out_array.attrs)
            self.assertEqual(right_attrs.name, out_array.name)
            self.assertEqual(id(self.grid), id(out_array.lawr.grid))

    def test_to_z_from_rate_value_check(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('rr')

        right_array = self.array.lawr.zr_convert(inverse=True)

        out_array = self.array.lawr.to_z()
        xr.testing.assert_identical(right_array, out_array)

    def test_to_z_from_decibel_rate_value_check(self):
        self.array = self.array.lawr.set_variable('dbrr')

        right_array = self.array.lawr.db_to_linear().lawr.zr_convert(
            inverse=True
        )

        out_array = self.array.lawr.to_z()
        xr.testing.assert_identical(right_array, out_array)

    def test_to_z_from_refl_value_check(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('dbz')

        right_array = self.array.lawr.db_to_linear()

        out_array = self.array.lawr.to_z()
        xr.testing.assert_identical(right_array, out_array)

    def test_to_z_from_lin_refl_value_check(self):
        self.array = self.array.lawr.set_variable('z')

        out_array = self.array.lawr.to_z()
        xr.testing.assert_identical(self.array, out_array)

    def test_to_z_without_name_assumes_lin_refl(self):
        self.array = self.array.lawr.set_variable('dbz')
        right_array = self.array.lawr.to_z()

        self.array.name = None
        out_array = self.array.lawr.to_z()

        xr.testing.assert_identical(right_array, out_array)

    def test_to_z_zero_value_check(self):
        self.array.values.fill(-32.5)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('dbz')
        out_array = self.array.lawr.to_z()
        self.assertEqual(out_array.values[0, 0, 0], 0.)

    def test_to_z_relative_negative_value_check(self):
        self.array.values.fill(-40.)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('dbz')
        out_array = self.array.lawr.to_z()
        self.assertEqual(out_array.values[0, 0, 0], 0.)

    def test_to_z_nan_value_check(self):
        self.array.values.fill(np.nan)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('dbz')
        out_array = self.array.lawr.to_z()
        self.assertTrue(np.isnan(out_array.values[0, 0, 0]))

    def test_to_dbz_zero_value_check(self):
        self.array.values.fill(0.)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('z')
        out_array = self.array.lawr.to_dbz()
        self.assertEqual(out_array.values[0, 0, 0], -32.5)

    def test_to_dbz_negative_value_check(self):
        self.array.values.fill(-1.)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('z')
        out_array = self.array.lawr.to_dbz()
        self.assertEqual(out_array.values[0, 0, 0], -32.5)

    def test_to_dbz_nan_value_check(self):
        self.array.values.fill(np.nan)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.array = self.array.lawr.set_variable('z')
        out_array = self.array.lawr.to_dbz()
        self.assertTrue(np.isnan(out_array.values[0, 0, 0]))

    def test_to_dbz_same_as_to_z_with_decibel(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        trafo_vars = list(from_decibel_to_linear.keys()) + \
                     list(from_decibel_to_linear.values())
        for var in trafo_vars:
            tmp_array = self.array.lawr.set_variable(var)
            right_array = tmp_array.lawr.to_z().lawr.db_to_linear(inverse=True)

            out_array = tmp_array.lawr.to_dbz()

            xr.testing.assert_identical(right_array, out_array)

    def test_set_metadata_replaces_metadata_from_other_array(self):
        tmp_array = deepcopy(self.array)
        tmp_array.attrs = {}
        self.assertFalse(tmp_array.attrs)
        tmp_array = tmp_array.lawr.set_metadata(self.array)
        self.assertDictEqual(tmp_array.attrs, self.array.attrs)

    def test_set_metadata_sets_grid_if_set(self):
        tmp_array = deepcopy(self.array)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.assertIsNone(tmp_array.lawr._grid)
        tmp_array = tmp_array.lawr.set_metadata(self.array)
        self.assertIsNotNone(tmp_array.lawr._grid)
        self.assertEqual(id(self.grid), id(tmp_array.lawr.grid))

    def test_set_name(self):
        tmp_array = deepcopy(self.array)
        tmp_array.name = None
        self.array.name = 'bla'
        self.assertIsNone(tmp_array.name)
        tmp_array = tmp_array.lawr.set_metadata(self.array)
        self.assertIsNotNone(tmp_array.name)
        self.assertEqual(tmp_array.name, self.array.name)


if __name__ == '__main__':
    unittest.main()
