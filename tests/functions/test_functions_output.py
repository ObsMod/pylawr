#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from tempfile import NamedTemporaryFile

# External modules
import xarray as xr

# Internal modules
from pylawr.functions.output import save_netcdf
from pylawr.grid import PolarGrid
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestSaveToNCFunction(unittest.TestCase):
    def setUp(self):
        self.array = create_array(PolarGrid())
        with NamedTemporaryFile() as f:
            self.test_path = f.name

    def tearDown(self):
        if os.path.isfile(self.test_path):
            os.remove(self.test_path)
        self.array.close()

    def test_saves_as_netcdf(self):
        self.assertFalse(os.path.isfile(self.test_path))
        save_netcdf(self.array, self.test_path)
        self.assertTrue(os.path.isfile(self.test_path))
        load_array = xr.load_dataarray(self.test_path)
        xr.testing.assert_identical(self.array, load_array)

    @unittest.expectedFailure
    def test_saves_as_netcdf_encoding(self):
        self.assertFalse(os.path.isfile(self.test_path))
        save_netcdf(self.array, self.test_path,
                    encoding={'time': {'units': 'days since 1970-01-01'}})
        self.assertTrue(os.path.isfile(self.test_path))
        load_array = xr.load_dataarray(self.test_path)
        xr.testing.assert_identical(self.array, load_array)



if __name__ == '__main__':
    unittest.main()
