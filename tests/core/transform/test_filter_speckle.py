#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import os
import unittest
import logging
from unittest.mock import patch, MagicMock

# External modules
import numpy as np

# Internal modules
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.filter.speckle import SpeckleFilter
from pylawr.utilities.helpers import create_array, polar_padding
from pylawr.grid.polar import PolarGrid

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestSpeckleFilter(unittest.TestCase):
    def setUp(self):
        self.array = create_array(PolarGrid(), const_val=-32.5)
        self.speckle = SpeckleFilter()

    def test_create_cluttermap_with_addname(self):
        addname = '_test'
        name = str(self.speckle.__class__.__name__) + addname
        cluttermap = self.speckle.create_cluttermap(array=self.array,
                                                    addname=addname)
        self.assertIn(name, cluttermap.layers.keys())

    def test_speckle_filter_calc_map_sets_map(self):
        self.assertIsNone(self.speckle.map)
        self.speckle.calc_map(self.array)
        self.assertIsNotNone(self.speckle.map)
        self.assertIsInstance(self.speckle.map, np.ndarray)

    def test_speckle_filter_sets_everything_to_zero(self):
        self.speckle.calc_map(self.array)
        np.testing.assert_equal(self.speckle.map, 0)

    def test_speckle_filter_sets_rain_to_nonzero(self):
        self.array[..., 100, 100] = 10
        self.speckle.calc_map(self.array)
        with self.assertRaises(AssertionError):
            np.testing.assert_equal(self.speckle.map, 0)
        self.assertEqual(self.speckle.map[100, 100], 8)

    def test_speckle_filter_sets_rain_under_5dbz_to_no_rain(self):
        self.array[:] = 4.9999999
        self.speckle.calc_map(self.array)
        np.testing.assert_equal(self.speckle.map, 0)

    def test_speckle_filter_uses_polar_zero_padding(self):
        self.speckle.window_size = (7, 7)
        rain_array = np.zeros_like(self.array.values.squeeze())
        padded_rain = polar_padding(rain_array, (3, 3))
        with patch('pylawr.transform.filter.speckle.polar_padding',
                   spec=np.ndarray, return_value=padded_rain) as pad_patch:
            self.speckle.calc_map(self.array)
        pad_patch.assert_called_once()
        self.assertListEqual(pad_patch.call_args[0][1], [3, 3])

    def test_speckle_filter_cluttermap_calls_calc_map_if_array(self):
        self.speckle.calc_map(self.array)
        self.speckle.calc_map = MagicMock()
        _ = self.speckle.create_cluttermap(self.array)
        self.speckle.calc_map.assert_called_once_with(self.array)

    def test_speckle_filter_cmap_returns_cmap(self):
        cmap = self.speckle.create_cluttermap(self.array)
        self.assertIsInstance(cmap, ClutterMap)
        self.assertEqual(list(cmap.layers.keys())[0], 'SpeckleFilter')

    def test_speckle_filter_cmap_sets_non_clutter_to_zero(self):
        cmap = self.speckle.create_cluttermap(self.array)
        cmap_array = cmap.layers['SpeckleFilter']
        np.testing.assert_equal(cmap_array, 0)

    def test_speckle_filter_cmap_sets_speckle_to_one(self):
        self.array[..., 100, 100] = 10
        cmap = self.speckle.create_cluttermap(self.array)
        cmap_array = cmap.layers['SpeckleFilter']
        self.assertEqual(cmap_array[..., 100, 100], 1)
        np.testing.assert_allclose(cmap_array, 0, atol=1)


if __name__ == '__main__':
    unittest.main()
