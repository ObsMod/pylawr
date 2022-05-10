#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import datetime
import io
import logging
import os
import warnings
import unittest
from unittest.mock import patch, PropertyMock

# External modules
import numpy as np
import pytz
import xarray as xr

# Internal modules
from pylawr.datahandler import LawrHandler
from pylawr.utilities.conventions import naming_convention

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL',logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class Unseekable(io.StringIO):
    def seek(self, *args, **kwargs):
        raise io.UnsupportedOperation


class TestLawrHandler(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(DATA_PATH, 'lawr_data.txt')
        self.file_handler = open(self.path, mode='r')
        self.data_handler = LawrHandler(self.file_handler)
        self.date = datetime.datetime(2017, 7, 20, 8, 46, 30)

    def tearDown(self):
        self.file_handler.close()

    def test_file_property_returns_private_file_if_not_none(self):
        self.data_handler._file = 'bla'
        self.assertEqual(id(self.data_handler.file),
                         id(self.data_handler._file))

    def test_file_property_returns_stringio_if_none(self):
        self.data_handler._file = None
        self.assertIsInstance(self.data_handler.file, io.StringIO)

    def test_file_property_decodes_bytes(self):
        self.data_handler._file = None
        fh = open(self.path, 'rb')
        logging.info(type(fh))
        text = fh.read().decode('utf-8')
        fh.seek(0)
        dh = LawrHandler(fh)
        returned_text = dh.file.read()
        fh.close()
        self.assertEqual(text, returned_text)

    def test_file_not_error_raise_if_unseekable(self):
        self.data_handler._file = None
        data = self.file_handler.read()
        unseekable = Unseekable(data)
        dh = LawrHandler(unseekable)
        _ = dh.file
        unseekable.close()

    def test_file_property_reads_filehandler_and_decode_fh(self):
        self.data_handler._file = None
        text = self.data_handler._fh.read()
        self.data_handler._fh.seek(0)
        str_value = self.data_handler.file.getvalue()
        self.assertEqual(text, str_value)

    def test_file_property_seeks_file(self):
        self.data_handler._file = None
        str_value = self.data_handler.file.getvalue()
        text = self.data_handler._fh.read()
        self.assertEqual(text, str_value)

    def test_decode_header_returns_dict(self):
        header_str = self.data_handler.file.readline()
        returned_header = self.data_handler._decode_header(header=header_str)
        self.assertIsInstance(returned_header, dict)

    def test_decode_header_contains_datetime(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        aware_dt = datetime.datetime.strptime(
            splitted_header[1], '%y%m%d%H%M%S')
        tzinfo = pytz.timezone(splitted_header[2])
        dt_header = tzinfo.localize(aware_dt).astimezone(pytz.utc).\
            replace(tzinfo=None)
        returned_header = self.data_handler._decode_header(header=header_str)
        self.assertIn('datetime', returned_header.keys())
        self.assertEqual(returned_header['datetime'], dt_header)

    def test_decode_header_contains_radartype(self):
        header_str = self.data_handler.file.readline()
        returned_header = self.data_handler._decode_header(header=header_str)
        self.assertIn('radartype', returned_header.keys())
        self.assertEqual(returned_header['radartype'], 'LAWR')

    def test_decode_header_contains_keyval(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        keyval_dict = self.data_handler._decode_keyval(splitted_header)
        header_dict = self.data_handler._decode_header(header=header_str)
        self.assertTrue(keyval_dict.items() <= header_dict.items())

    def test_prepare_header_returns_list(self):
        header_str = self.data_handler.file.readline()
        returned_header = self.data_handler._prepare_header(header=header_str)
        self.assertIsInstance(returned_header, list)

    def test_prepare_header_splits_header(self):
        header_str = self.data_handler.file.readline()
        returned_header = self.data_handler._prepare_header(header=header_str)
        splitted_header = header_str.split()
        self.assertListEqual(returned_header, splitted_header)

    def test_decode_datetime_returns_datetime(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        returned_datetime = self.data_handler._decode_datetime(
            *splitted_header[1:3])
        self.assertIsInstance(returned_datetime, datetime.datetime)

    def test_decode_datetime_returns_right_datetime(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        returned_datetime = self.data_handler._decode_datetime(
            *splitted_header[1:3])
        aware_dt = datetime.datetime.strptime(
            splitted_header[1], '%y%m%d%H%M%S')
        tzinfo = pytz.timezone(splitted_header[2])
        dt_header = tzinfo.localize(aware_dt).astimezone(pytz.utc)
        dt_header = dt_header.replace(tzinfo=None)
        self.assertEqual(returned_datetime, dt_header)

    def test_decode_datetime_uses_tz_str(self):
        dt_str = '200101020000'
        tz_str = 'Etc/GMT+2'
        returned_datetime = self.data_handler._decode_datetime(dt_str, tz_str)
        aware_dt = datetime.datetime.strptime(
            dt_str, '%y%m%d%H%M%S')
        tzinfo = pytz.timezone(tz_str)
        dt_header = tzinfo.localize(aware_dt).astimezone(pytz.utc)
        dt_header = dt_header.replace(tzinfo=None)
        self.assertEqual(returned_datetime, dt_header)

    def test_decode_datetime_has_no_timezone(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        returned_datetime = self.data_handler._decode_datetime(
            *splitted_header[1:3])
        self.assertIsNone(returned_datetime.tzinfo)

    def test_decode_datetime_no_timezone(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        try:
            returned_datetime = self.data_handler._decode_datetime(
                splitted_header[1], None)
        except AttributeError as e:
            self.fail('AttributeError was raised\n{0}'.format(e))
        aware_dt = datetime.datetime.strptime(
            splitted_header[1], '%y%m%d%H%M%S')
        self.assertEqual(returned_datetime, aware_dt)

    def test_decode_keyval_returns_dict(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        decoded_keyval = self.data_handler._decode_keyval(splitted_header)
        self.assertIsInstance(decoded_keyval, dict)

    def test_decode_keyval_matches_key_val(self):
        header_str = self.data_handler.file.readline()
        splitted_header = self.data_handler._prepare_header(header_str)
        right_dict = {
            splitted_header[k-1]: float(splitted_header[k+1])
            for k, sub in enumerate(splitted_header)
            if sub == '=' and k > 0 and k+1 < len(splitted_header)
        }
        returned_dict = self.data_handler._decode_keyval(splitted_header)
        self.assertDictEqual(returned_dict, right_dict)

    def test_decode_data_returns_numpy_array(self):
        _ = self.data_handler.file.readline()
        raw_data = self.data_handler.file.readlines()
        decoded_data = self.data_handler._decode_data(raw_data)
        self.assertIsInstance(decoded_data, np.ndarray)

    def test_decode_data_returns_data_array(self):
        _ = self.data_handler.file.readline()
        raw_data = self.data_handler.file.readlines()
        right_array = np.array([l.split()[1:] for l in raw_data],
                               dtype=np.float32)
        decoded_data = self.data_handler._decode_data(raw_data)
        np.testing.assert_almost_equal(decoded_data, right_array)
        self.assertTupleEqual(decoded_data.shape, (360, 333))

    def test_prepare_dataline_returns_list(self):
        _ = self.data_handler.file.readline()
        raw_data = self.data_handler.file.readline()
        prepared_line = self.data_handler._prepare_dataline(raw_data)
        self.assertIsInstance(prepared_line, list)

    def test_prepare_dataline_splits_dataline(self):
        _ = self.data_handler.file.readline()
        raw_data = self.data_handler.file.readline()
        prepared_line = self.data_handler._prepare_dataline(raw_data)
        self.assertListEqual(prepared_line, raw_data.split())

    def test_header_property_returns_header_and_sets_attribute(self):
        self.data_handler._header = None
        header_text = self.data_handler.file.readline()
        decoded_header = self.data_handler._decode_header(header_text)
        self.assertEqual(decoded_header, self.data_handler.header)
        self.assertEqual(self.data_handler._header, decoded_header)

    def test_header_property_returns_saved_header(self):
        self.data_handler._header = {'test': 'bla'}
        self.assertEqual(self.data_handler._header, self.data_handler.header)

    def test_get_available_dates_extract_date_from_header(self):
        extracted_date = self.data_handler.header['datetime']
        self.assertSequenceEqual([extracted_date, ],
                                 self.data_handler._get_available_dates())

    def test_available_dates_extract_date_from_header(self):
        self.data_handler._available_dates = None
        extracted_date = self.data_handler.header['datetime']
        self.assertSequenceEqual([extracted_date, ],
                                 self.data_handler.available_dates)
        self.assertEqual(self.data_handler._available_dates, [extracted_date, ])

    def test_available_dates_returns_saved_dates(self):
        self.data_handler._header = {'test': 'test'}
        self.assertEqual(self.data_handler._header, self.data_handler.header)

    def test_get_reflectivityreturns_dataarray(self):
        returned_array = self.data_handler.get_reflectivity()
        self.assertIsInstance(returned_array, xr.DataArray)

    def test_get_reflectivitysets_header_as_attributes(self):
        raw_header = self.data_handler.file.readline()
        decoded_header = self.data_handler._decode_header(raw_header)
        returned_array = self.data_handler.get_reflectivity()
        self.assertTrue(
            {k: decoded_header[k] for k in decoded_header.keys()
             if k != 'datetime'}.items() <=
            dict(returned_array.attrs).items(),
        )

    def test_get_reflectivity_sets_values_as_datavalues(self):
        _ = self.data_handler.file.readline()
        raw_data = self.data_handler.file.readlines()
        decoded_data = self.data_handler._decode_data(raw_data)
        returned_array = self.data_handler.get_reflectivity()
        np.testing.assert_equal(returned_array.values.squeeze(), decoded_data)

    def test_get_reflectivity_returns_correct_dataarray(self):
        raw_header = self.data_handler.file.readline()
        raw_data = self.data_handler.file.readlines()
        decoded_header = self.data_handler._decode_header(raw_header)
        decoded_data = self.data_handler._decode_data(raw_data)
        azi_len, range_len = decoded_data.shape
        coordinates = dict(
            time=(['time', ],
                  np.array((decoded_header['datetime'], ),
                           dtype='datetime64[ns]'),
                  naming_convention['time']),
            azimuth=(['azimuth', ], np.arange(azi_len),
                     naming_convention['azimuth']),
            range=(['range', ], np.arange(range_len),
                   naming_convention['range'])
        )
        attrs = dict(unit='dBZ')
        attrs.update({k: decoded_header[k] for k in decoded_header.keys()
                      if k != 'datetime'})
        attrs.update(naming_convention['dbz'])
        right_dataarray = xr.DataArray(
            data=decoded_data[np.newaxis, ...],
            coords=coordinates,
            dims=['time', 'azimuth', 'range'],
            attrs=attrs,
            name='dbz'
        )
        returned_array = self.data_handler.get_reflectivity()
        xr.testing.assert_identical(returned_array, right_dataarray)

    def test_get_reflectivity_seeks_file_position(self):
        right_array = self.data_handler.get_reflectivity()
        self.data_handler.file.seek(20)
        returned_array = self.data_handler.get_reflectivity()
        xr.testing.assert_identical(returned_array, right_array)

    def test_get_reflectivity_doesnt_raise_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as warn:
            warnings.simplefilter("always")
            _ = self.data_handler.get_reflectivity()
            deprecation_sub = [issubclass(w.category, DeprecationWarning)
                               for w in warn]
            self.assertFalse(any(deprecation_sub))

    @patch('pylawr.datahandler.lawr.LawrHandler._decode_data',
           **{'return_value.raiseError.side_effect': Exception()})
    def test_get_reflectivity_error_with_right_message_if_corrupted(self,
                                                                    fmock):
        fh = open(os.path.join(DATA_PATH, 'lawr_data.txt'), 'r')
        self.data_handler = LawrHandler(fh)
        error_message = 'The given file is corrupted and couldn\'t be ' \
                        'decoded!'
        with self.assertRaises(ValueError) as err:
            _ = self.data_handler.get_reflectivity()
        self.assertEqual(str(err.exception), error_message)
        fh.close()

    @patch('pylawr.datahandler.lawr.LawrHandler._decode_header',
           return_value={"n_p": 360+317})
    def test_reflectivity_error_if_missing_azimuth(self,
                                                   fmock):
        fh = open(os.path.join(DATA_PATH, 'lawr_data.txt'), 'r')
        self.data_handler = LawrHandler(fh)
        error_message = 'There are {0:d} missing azimuth angles within the ' \
                        'file!'.format(317)
        with self.assertRaises(ValueError) as err:
            _ = self.data_handler.get_reflectivity()
        self.assertEqual(str(err.exception), error_message)
        fh.close()

    def test_reflectivity_sets_attr_based_on_convention(self):
        var = 'dbz'
        returned_array = self.data_handler.get_reflectivity()
        attrs = {a: returned_array.attrs[a] for a in naming_convention[var]}
        self.assertDictEqual(attrs, naming_convention[var])

    def test_reflectivity_sets_name_to_reflectivity(self):
        var = 'dbz'
        returned_array = self.data_handler.get_reflectivity()
        self.assertEqual(var, returned_array.name)

    def test_get_reflectivity_set_attrs_following_conventions(self):
        returned_array = self.data_handler.get_reflectivity()
        self.assertTrue(naming_convention[returned_array.name].items() <=
                        returned_array.attrs.items())
        for dim in returned_array.dims[1:]:
            self.assertTrue(naming_convention[dim].items()
                            <= returned_array[dim].attrs.items())


if __name__ == '__main__':
    unittest.main()
