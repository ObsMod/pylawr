#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import logging
import unittest
import os
from collections import OrderedDict
from unittest.mock import patch

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.grid import PolarGrid
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.grid import PolarGrid
from pylawr.utilities.helpers import create_array

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class ClutterMapTest(unittest.TestCase):
    def setUp(self):
        self.array = create_array(PolarGrid(), tag=None)

        self.name1 = "TestFilter1"
        self.name2 = "TestFilter2"
        self.values1 = np.random.randint(2, size=self.array.shape)
        self.values2 = np.random.randint(2, size=self.array.shape)

        self.TCM1 = ClutterMap(self.name1, self.values1)
        self.TCM2 = ClutterMap(self.name2, self.values2)

    def test_append_append_values(self):
        dict_expect = OrderedDict()
        dict_expect[self.name1] = self.values1
        dict_expect[self.name2] = self.values2
        self.TCM1.append(self.TCM2)
        returned_layer_dict = self.TCM1.layers
        self.assertDictEqual(returned_layer_dict, dict_expect)

    def test_fuzzy_threshold(self):
        self.assertIsNotNone(self.TCM1.fuzzy_threshold)

    def test_transform_onelayer_fuzzyone(self):
        transformed_array = self.TCM1.transform(self.array)
        self.assertEqual(np.count_nonzero(np.isnan(transformed_array.values)),
                         np.count_nonzero(self.values1 == 1))

    def test_transform_twolayer_all(self):
        self.TCM1.append(self.TCM2)
        self.TCM1.fuzzy_threshold = 1

        transformed_array = self.TCM1.transform(self.array)

        test_clutter = np.logical_and(self.values1, self.values2)

        self.assertEqual(np.count_nonzero(np.isnan(transformed_array.values)),
                         np.count_nonzero(test_clutter == 1))

    def test_transform_twolayer_any(self):
        self.TCM1.append(self.TCM2)
        self.TCM1.fuzzy_threshold = 0 

        transformed_array = self.TCM1.transform(self.array)

        test_clutter = np.logical_or(self.values1, self.values2)

        self.assertEqual(np.sum(np.isnan(transformed_array.values)),
                         np.sum(test_clutter))

    def test_constructor_empty(self):
        none_clt = ClutterMap()

        self.assertTrue(not none_clt.layers)

    def test_merge_empty(self):
        none_clt = ClutterMap()

        none_clt.append(self.TCM1)

        self.assertFalse(not none_clt.layers)

    def test_transform_empty(self):
        none_clt = ClutterMap()

        transformed_array = none_clt.transform(self.array)

        np.testing.assert_array_equal(transformed_array, self.array)

    def test_none_cmap(self):
        none_clt = ClutterMap("None")

        self.assertEqual(len(none_clt.layers), 0)

    def test_array_returns_array_with_layer_values(self):
        self.TCM1.append(self.TCM2)
        right_array = np.concatenate([self.values1, self.values2])
        returned_array = self.TCM1.array

        np.testing.assert_equal(returned_array, right_array)

    def test_array_raise_value_error_if_no_layer(self):
        cluttermap = ClutterMap()
        with self.assertRaises(ValueError) as e:
            _ = cluttermap.array

    def test_append_appends_weights(self):
        right_weights = OrderedDict()
        right_weights[self.name1] = 1
        right_weights[self.name2] = 2
        self.TCM2.weights[self.name2] = right_weights[self.name2]
        self.TCM1.append(self.TCM2)
        returned_weights = self.TCM1.weights
        self.assertDictEqual(right_weights, returned_weights)

    def test_append_raises_value_error_if_added_cmap_has_not_right_shape(self):
        self.TCM2.layers[self.name2] = self.TCM2.layers[self.name2][..., :1]
        with self.assertRaises(ValueError):
            self.TCM1.append(self.TCM2)

    def test_mean_returns_weighted_mean_of_array(self):
        self.TCM2.weights[self.name2] = 2
        self.TCM1.append(self.TCM2)
        right_average = np.average(self.TCM1.array, axis=0, weights=[1, 2])
        returned_average = self.TCM1.mean()
        np.testing.assert_equal(returned_average, right_average)

    def test_str_returns_set_str(self):
        self.TCM2.weights[self.name2] = 2
        self.TCM1.append(self.TCM2)
        right_str = 'ClutterMap(1*{0:s}, 2*{1:s})'.format(self.name1,
                                                          self.name2)
        self.assertEqual(str(self.TCM1), right_str)

    def test_transform_calls_mean(self):
        self.TCM2.weights[self.name2] = 2
        self.TCM1.append(self.TCM2)
        mean_values = self.TCM1.mean()
        with patch('pylawr.transform.filter.cluttermap.ClutterMap.mean',
                   return_value=mean_values) as p:
            _ = self.TCM1.transform(array=self.array)
            p.assert_called_once_with()

    def test_transform_tags_array(self):
        self.TCM2.weights[self.name2] = 2
        self.TCM1.append(self.TCM2)
        returned_array = self.TCM1.transform(self.array)
        right_tag = 'filtered with {0:s}'.format(str(self.TCM1))
        self.assertEqual(right_tag, returned_array.attrs['tags'])

    def test_transform_sets_grid_coordinates(self):
        self.array = self.array.lawr.set_grid_coordinates(PolarGrid())
        returned_array = self.TCM1.transform(self.array)
        self.assertEqual(id(self.array.lawr.grid), id(returned_array.lawr.grid))
        xr.testing.assert_equal(self.array['azimuth'],
                                returned_array['azimuth'])
        xr.testing.assert_equal(self.array['range'], returned_array['range'])


if __name__ == '__main__':
    unittest.main()
