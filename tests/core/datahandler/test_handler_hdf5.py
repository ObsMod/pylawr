#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
from unittest.mock import patch
import logging
import os
import datetime
from copy import deepcopy

# External modules
import h5py
import numpy as np
import pytz
import xarray as xr

# Internal modules
from pylawr.datahandler.hdf5 import DWDHDF5Handler
from pylawr.grid import PolarGrid, GridNotAvailableError
from pylawr.utilities.conventions import naming_convention


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestHDF5Handler(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(DATA_PATH, 'dwd_test_data.hdf5')
        self.file_handler = open(self.path, mode='rb')
        self.data_handler = DWDHDF5Handler(self.file_handler)
        self.date = datetime.datetime(2017, 11, 6, 0, 10, 3, tzinfo=pytz.UTC)

    def tearDown(self):
        self.file_handler.close()

    def test_file_property_returns_private_file_if_not_none(self):
        self.data_handler._file = 'bla'
        self.assertEqual(id(self.data_handler.file),
                         id(self.data_handler._file))

    def test_file_property_returns_h5py_file_if_none(self):
        self.data_handler._file = None
        self.assertIsInstance(self.data_handler.file, h5py.File)

    def test_file_gives_temp_file_to_hdf5_with_same_content(self):
        right_hdf5_file = h5py.File(self.path, mode='r')
        returned_hdf5_file = self.data_handler.file
        np.testing.assert_equal(
            np.array(right_hdf5_file['dataset1']['data1']['data']),
            np.array(returned_hdf5_file['dataset1']['data1']['data']),
        )

    def test_file_reset_file_handler_to_unread(self):
        _ = self.file_handler.read(2)
        try:
            returned_hdf5_file = self.data_handler.file
        except OSError:
            raise AssertionError('HDF5 file is not reset!')

    def test_get_datetime_returns_datetime64(self):
        dataset = self.data_handler.file['dataset1']
        returned_datetime = self.data_handler._decode_datetime_from_node(
            dataset
        )
        self.assertIsInstance(returned_datetime, np.datetime64)

    def test_get_datetime_decodes_datetime_from_data(self):
        dataset = self.data_handler.file['dataset1']
        returned_datetime = self.data_handler._decode_datetime_from_node(
            dataset
        )
        true_datetime = np.datetime64(
            self.date.astimezone(pytz.UTC).replace(tzinfo=None))
        self.assertEqual(true_datetime, returned_datetime)

    def test_get_available_dates_returns_list_of_date(self):
        dataset = self.data_handler.file['dataset1']
        returned_datetime = self.data_handler._decode_datetime_from_node(
            dataset
        )
        avail_dates = [returned_datetime, ]
        self.assertListEqual(
            self.data_handler._get_available_dates(), avail_dates
        )

    def test_get_center_returns_lat_lon_tuple(self):
        lat = self.data_handler.file['where'].attrs['lat']
        lon = self.data_handler.file['where'].attrs['lon']
        height = self.data_handler.file['where'].attrs['height']
        right_lat_lon = (lat, lon, height)
        returned_lat_lon = self.data_handler._get_center()
        self.assertTupleEqual(right_lat_lon, returned_lat_lon)

    def test_get_grid_data_can_be_set_for_grid(self):
        dataset = self.data_handler.file['dataset1']
        grid_data = self.data_handler._get_grid_data(dataset)
        self.assertIsInstance(grid_data, dict)
        _ = PolarGrid(**grid_data)

    def test_get_grid_raises_grid_not_available_error_if_not_available(self):
        dataset = self.data_handler.file
        with self.assertRaises(GridNotAvailableError):
            _ = self.data_handler._get_grid_data(dataset)

    def test_grid_uses_grid_data_and_creates_polargrid(self):
        dataset = self.data_handler.file['dataset1']
        grid_data = self.data_handler._get_grid_data(dataset)
        right_grid = PolarGrid(**grid_data)
        output_grid = self.data_handler.grid
        self.assertIsInstance(output_grid, PolarGrid)
        np.testing.assert_equal(
            np.concatenate(output_grid.coords),
            np.concatenate(right_grid.coords),
        )

    def test_check_version_product_checks_object(self):
        with h5py.File(self.path, mode='r+') as hdf5_file:
            hdf5_file['what'].attrs['object'] = b'RAY'
            try:
                self.assertFalse(self.data_handler.hdf5_validation(hdf5_file))
                hdf5_file['what'].attrs['object'] = b'PVOL'
                self.assertTrue(self.data_handler.hdf5_validation(hdf5_file))
            except AssertionError as e:
                raise e
            finally:
                hdf5_file['what'].attrs['object'] = b'PVOL'

    def test_check_version_product_checks_version(self):
        with h5py.File(self.path, mode='r+') as hdf5_file:
            hdf5_file['what'].attrs['version'] = b'H5RAD 1.0'
            try:
                self.assertFalse(self.data_handler.hdf5_validation(hdf5_file))
                hdf5_file['what'].attrs['version'] = b'H5RAD 2.2'
                self.assertTrue(self.data_handler.hdf5_validation(hdf5_file))
            except AssertionError as e:
                raise e
            finally:
                hdf5_file['what'].attrs['version'] = b'H5RAD 2.2'

    @patch('pylawr.datahandler.hdf5.DWDHDF5Handler.hdf5_validation',
           return_value=False)
    def test_file_checks_version_product_raises_fileerror(self, *args):
        err_msg = 'This hdf5 data handler is not configured to load this data!'
        with self.assertRaises(ValueError) as e:
            _ = self.data_handler.file
        self.assertEqual(str(e.exception), err_msg)

    def test_get_metadata_from_node_extracts_metadata_from_dataset(self):
        dataset = self.data_handler.file['dataset1']
        right_metadata = dict(dataset['where'].attrs)
        right_metadata.update(dict(dataset['what'].attrs))
        right_metadata.update(dict(dataset['how'].attrs))
        right_metadata = {k: v for k, v in right_metadata.items()
                          if not isinstance(v, np.ndarray)}
        right_metadata = {k: v.decode('UTF-8') if isinstance(v, bytes) else v
                          for k, v in right_metadata.items()}
        returned_metadata = self.data_handler._get_metadata_from_node(dataset)
        self.assertDictEqual(right_metadata, returned_metadata)
        xr.DataArray(data=[], attrs=returned_metadata).to_netcdf('test.nc')
        os.remove('test.nc')

    def test_get_metadata_from_node_extracts_metadata_from_data(self):
        data = self.data_handler.file['dataset1']['data1']
        right_metadata = dict(data['what'].attrs)
        right_metadata = {k: v for k, v in right_metadata.items()
                          if not isinstance(v, np.ndarray)}
        right_metadata = {k: v.decode('UTF-8') if isinstance(v, bytes) else v
                          for k, v in right_metadata.items()}
        returned_metadata = self.data_handler._get_metadata_from_node(data)
        self.assertDictEqual(right_metadata, returned_metadata)
        xr.DataArray(data=[], attrs=returned_metadata).to_netcdf('test.nc')
        os.remove('test.nc')

    def test_get_metadata_from_node_extracts_metadata_from_root(self):
        data = self.data_handler.file
        right_metadata = dict(data['where'].attrs)
        right_metadata.update(dict(data['what'].attrs))
        right_metadata.update(dict(data['how'].attrs))
        right_metadata = {k: v for k, v in right_metadata.items()
                          if not isinstance(v, np.ndarray)}
        right_metadata = {k: v.decode('UTF-8') if isinstance(v, bytes) else v
                          for k, v in right_metadata.items()}
        returned_metadata = self.data_handler._get_metadata_from_node(data)
        self.assertDictEqual(right_metadata, returned_metadata)
        xr.DataArray(data=[], attrs=returned_metadata).to_netcdf('test.nc')
        os.remove('test.nc')

    def test_get_dataset_from_node_returns_xr_dataset(self):
        node = self.data_handler.file['dataset1']
        returned_dataset = self.data_handler._get_dataset_from_node(node)
        self.assertIsInstance(returned_dataset, xr.Dataset)

    def test_get_dataset_from_node_sets_metadata_from_node(self):
        node = self.data_handler.file['dataset1']
        metadata = self.data_handler._get_metadata_from_node(node)
        returned_dataset = self.data_handler._get_dataset_from_node(node)
        self.assertDictEqual(returned_dataset.attrs, metadata)

    def test_get_grid_from_node_returns_polar_grid_from_dataset(self):
        node = self.data_handler.file['dataset1']
        returned_grid = self.data_handler._get_grid_from_node(node)
        right_grid = PolarGrid(**self.data_handler._get_grid_data(node))
        np.testing.assert_equal(
            np.concatenate(returned_grid.coords),
            np.concatenate(right_grid.coords),
        )

    def test_get_grid_returns_none_if_grid_not_available_error(self):
        node = self.data_handler.file['dataset1']['data1']
        returned_grid = self.data_handler._get_grid_from_node(node)
        self.assertIsNone(returned_grid)

    def test_get_dataarray_from_node_returns_xr_dataarray(self):
        node = self.data_handler.file['dataset1']['data1']
        returned_da = self.data_handler._get_dataarray_from_node(node)
        self.assertIsInstance(returned_da, xr.DataArray)

    def test_get_dataarray_from_node_sets_metadata_from_node(self):
        node = self.data_handler.file['dataset1']['data2']
        metadata = self.data_handler._get_metadata_from_node(node)
        returned_da = self.data_handler._get_dataarray_from_node(node)
        self.assertDictEqual(returned_da.attrs, metadata)

    def test_get_dataarray_from_node_sets_data_from_data(self):
        node = self.data_handler.file['dataset1']['data3']
        metadata = self.data_handler._get_metadata_from_node(node)
        data = np.array(node['data']) * metadata['gain'] + metadata['offset']
        returned_da = self.data_handler._get_dataarray_from_node(node)
        np.testing.assert_equal(returned_da.values, data)

    def test_get_dataarray_from_node_sets_name_to_quantity(self):
        node = self.data_handler.file['dataset1']['data4']
        metadata = self.data_handler._get_metadata_from_node(node)
        returned_da = self.data_handler._get_dataarray_from_node(node)
        self.assertEqual(returned_da.name, metadata['quantity'])

    def test_get_dataarray_from_node_sets_nodata_to_nan(self):
        node = self.data_handler.file['dataset1']['data1']
        metadata = self.data_handler._get_metadata_from_node(node)
        data = np.array(node['data']) * metadata['gain'] + metadata['offset']
        fake_metadata = deepcopy(metadata)
        fake_metadata['nodata'] = np.min(data)
        data[data == fake_metadata['nodata']] = np.nan
        trg = 'pylawr.datahandler.hdf5.DWDHDF5Handler._get_metadata_from_node'
        with patch(trg, return_value=fake_metadata) as p:
            returned_da = self.data_handler._get_dataarray_from_node(node)
        np.testing.assert_equal(returned_da.values, data)

    def test_get_dataarray_from_node_returns_right_dataarray(self):
        node = self.data_handler.file['dataset1']['data4']
        metadata = self.data_handler._get_metadata_from_node(node)
        data = np.array(node['data']) * metadata['gain'] + metadata['offset']
        right_da = xr.DataArray(data, attrs=metadata, name=metadata['quantity'])
        returned_da = self.data_handler._get_dataarray_from_node(node)
        xr.testing.assert_identical(right_da, returned_da)

    def test_set_data_coords_expand_time_dim_if_given(self):
        true_datetime = np.datetime64(
            self.date.astimezone(pytz.UTC).replace(tzinfo=None))
        node = self.data_handler.file['dataset1']['data1']
        dataarray = self.data_handler._get_dataarray_from_node(node)
        returned_dataarray = self.data_handler._define_data_coords(
            dataarray, valid_dt=true_datetime
        )
        right_dataarray = dataarray.expand_dims('time', axis=0)
        right_dataarray['time'] = [true_datetime, ]
        right_dataarray['time'].attrs = naming_convention['time']
        xr.testing.assert_identical(right_dataarray, returned_dataarray)

    def test_set_data_coords_set_grid_coords(self):
        node = self.data_handler.file['dataset1']['data1']
        dataarray = self.data_handler._get_dataarray_from_node(node)
        grid = PolarGrid(
            **self.data_handler._get_grid_data(
                self.data_handler.file['dataset1']
            )
        )
        right_dataarray = dataarray.lawr.set_grid_coordinates(grid)
        returned_dataarray = self.data_handler._define_data_coords(
            dataarray, grid=grid
        )
        xr.testing.assert_identical(right_dataarray, returned_dataarray)

    def test_set_data_coords_set_attrs(self):
        true_datetime = np.datetime64(
            self.date.astimezone(pytz.UTC).replace(tzinfo=None))
        node = self.data_handler.file['dataset1']['data1']
        dataarray = self.data_handler._get_dataarray_from_node(node)
        grid = PolarGrid(
            **self.data_handler._get_grid_data(
                self.data_handler.file['dataset1']
            )
        )
        returned_dataarray = self.data_handler._define_data_coords(
            dataarray, grid=grid, valid_dt=true_datetime
        )
        for dim in returned_dataarray.dims:
            self.assertTrue(naming_convention[dim].items()
                            <= returned_dataarray[dim].attrs.items())

    def test_get_datasets_returns_tuple_with_all_available_datasets(self):
        ds_name = 'dataset1'
        node = self.data_handler.file[ds_name]
        right_dataset = self.data_handler._get_dataset_from_node(node)
        returned_datasets = self.data_handler.get_datasets()
        self.assertIs(len(returned_datasets), 1)
        self.assertIsInstance(returned_datasets[ds_name], xr.Dataset)
        xr.testing.assert_equal(right_dataset, returned_datasets[ds_name])

    def test_get_reflectivity_returns_var_from_dataset_1(self):
        datasets = self.data_handler.get_datasets()
        right_dataarray = datasets['dataset1']['DBZH']
        right_dataarray = right_dataarray.lawr.set_variable('dbz')
        returned_dataarray = self.data_handler.get_reflectivity()
        xr.testing.assert_equal(returned_dataarray, right_dataarray)

    def test_get_reflectivity_sets_attrs_following_conventions(self):
        returned_dataarray = self.data_handler.get_reflectivity()
        self.assertTrue(naming_convention[returned_dataarray.name].items() <=
                        returned_dataarray.attrs.items())
        for dim in returned_dataarray.dims:
            self.assertTrue(naming_convention[dim].items()
                            <= returned_dataarray[dim].attrs.items())

    def test_get_reflectivity_appends_grid_if_available(self):
        returned_dataarray = self.data_handler.get_reflectivity()
        returned_grid = returned_dataarray.lawr.grid
        right_grid = self.data_handler.grid
        np.testing.assert_equal(
            np.concatenate(returned_grid.coords),
            np.concatenate(right_grid.coords),
        )


if __name__ == '__main__':
    unittest.main()
