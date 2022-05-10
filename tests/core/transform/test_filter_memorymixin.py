#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
from unittest import mock
import logging
import inspect
import datetime
import os

# External modules
import xarray as xr
import numpy as np
import unittest

# Internal modules
from pylawr.transform.memorymixin import MemoryMixin
from pylawr.grid import PolarGrid
from pylawr.utilities.helpers import create_array

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class MyFilter(MemoryMixin):
    def __init__(self, a=1, b=2, c=3):
        self.a = a
        self.b = b
        self.c = c

    def fit(self, *args, **kwargs):
        pass

    def to_xarray(self, *args, **kwargs):
        pass

    def fitted(self):
        pass


class TestFilterClass(unittest.TestCase):
    def test_cannot_instantiate_bare_filter(self):
        with self.assertRaises(TypeError):
            MemoryMixin()


class TestMyFilter(unittest.TestCase):
    def setUp(self):
        self.f = MyFilter()
        cls_params = inspect.getfullargspec(self.f.__init__).args
        params_dict = {n: getattr(self.f, n) for n in cls_params
                       if not n == 'self'}
        self.xr_ds = self.to_dataset(params_dict)
        self.array = create_array(PolarGrid())

    def test_set_xr_params_sets_params(self):
        old_a = self.f.a
        self.f.a = 2
        self.f.set_xr_params(self.xr_ds)
        self.assertEqual(self.f.a, old_a)

    def test_from_xarray(self):
        filter2 = MyFilter.from_xarray(self.xr_ds)
        self.assertIsInstance(filter2, MyFilter)

    def to_dataset(self, params):
        xr_vars = {
            p: (
                (), params[p]
            ) for p in params.keys()
        }
        ds = xr.Dataset(
            data_vars=xr_vars,
        )
        ds.attrs['type'] = self.__class__.__name__
        return ds

    def test_get_time_raises_typeerror(self):
        time_obj = self.array
        with self.assertRaises(TypeError) as e:
            _ = self.f._get_time(self.array, time_obj)
            logger.warning(str(e.msg))

    def test_get_time_uses_array_time_if_no_time_obj(self):
        time_obj = self.f._get_time(self.array, None)
        np.testing.assert_equal(
            np.asarray(self.array.time, dtype="datetime64[ns]"),
            time_obj
        )

    def test_get_time_uses_now_if_not_time(self):
        self.array = self.array.squeeze('time', drop=True)

        time_right_old = np.datetime64(datetime.datetime.now())
        time_out = self.f._get_time(self.array, None)
        time_right_new = np.datetime64(datetime.datetime.now())

        self.assertGreater(time_out, time_right_old)
        self.assertLess(time_out, time_right_new)
        self.assertIsInstance(time_out, np.datetime64)

    def test_get_time_uses_time_obj(self):
        time_obj = datetime.datetime.now()
        time_out = self.f._get_time(self.array, time_obj)
        time_right = np.asarray(
            time_obj, dtype="datetime64[ns]"
        )
        self.assertEqual(time_right, time_out)


if __name__ == '__main__':
    unittest.main()
