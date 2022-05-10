#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import os
import unittest
import logging
from unittest.mock import MagicMock

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.grid import PolarGrid
from pylawr.utilities.helpers import create_array
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.filter.temporal import TemporalFilter


logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

rnd = np.random.RandomState(42)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestTemporalFilter(unittest.TestCase):
    def setUp(self):
        self.array = create_array(PolarGrid())
        self.temporal = TemporalFilter()

    def test_create_cluttermap_with_addname(self):
        addname = '_test'
        name = str(self.temporal.__class__.__name__) + addname
        cluttermap = self.temporal.create_cluttermap(array=self.array,
                                                     addname=addname)
        self.assertIn(name, cluttermap.layers.keys())

    def test_fit_sets_hist_map_to_rain_map(self):
        self.assertIsNone(self.temporal._hist_maps)
        self.temporal.fit(self.array)
        self.assertIsNotNone(self.temporal._hist_maps)
        np.testing.assert_equal(self.temporal._hist_maps, 0)
        self.assertEqual(self.temporal._hist_maps.ndim, 3)

    def test_fit_sets_rain_to_one(self):
        self.array[..., 100, 100] = 10
        self.temporal.fit(self.array)
        np.testing.assert_allclose(self.temporal._hist_maps, 0, atol=1)
        self.assertEqual(self.temporal._hist_maps[..., 100, 100], 1)

    def test_fit_sets_rain_below_threshold_to_no_rain(self):
        self.array[..., 100, 100] = 4.999999
        self.temporal.fit(self.array)
        np.testing.assert_equal(self.temporal._hist_maps, 0)

    def test_fit_concatenate_hist_map_with_array(self):
        self.array[..., :, :] = 10
        self.temporal.fit(self.array)
        self.temporal.fit(self.array)
        np.testing.assert_equal(self.temporal._hist_maps, 1)
        self.assertEqual(len(self.temporal._hist_maps), 2)

    def test_fit_stores_only_the_n_last_maps(self):
        self.temporal.store_n_images = 5
        rand_vals = rnd.uniform(0, 10, size=10)
        for i in range(10):
            self.array[..., 100, 100] = rand_vals[i]
            self.temporal.fit(self.array)
        self.assertEqual(len(self.temporal._hist_maps), 5)
        np.testing.assert_equal(
            self.temporal._hist_maps[:, 100, 100], (rand_vals > 5)[-5:]
        )

    def test_fitted_checks_if_full_hist_maps(self):
        self.temporal.store_n_images = 3
        self.assertFalse(self.temporal.fitted)

        self.temporal.fit(self.array)
        self.assertFalse(self.temporal.fitted)

        self.temporal.fit(self.array)
        self.assertFalse(self.temporal.fitted)

        self.temporal.fit(self.array)
        self.assertTrue(self.temporal.fitted)

        self.temporal.fit(self.array)
        self.assertTrue(self.temporal.fitted)

    def test_calc_map_sets_map(self):
        self.assertFalse(self.temporal._map)
        self.temporal.calc_map(self.array)
        np.testing.assert_equal(self.temporal._map, 0)

    def test_calc_map_sets_rain_to_one(self):
        self.array[..., 100, 100] = 10
        self.temporal.calc_map(self.array)
        np.testing.assert_allclose(self.temporal._map, 0, atol=1)
        self.assertEqual(self.temporal._map[..., 100, 100], 1)

    def test_calc_map_sets_rain_below_threshold_to_no_rain(self):
        self.array[..., 100, 100] = 4.999999
        self.temporal.calc_map(self.array)
        np.testing.assert_equal(self.temporal._map, 0)

    def test_calc_map_calls_to_dbz(self):
        self.array.lawr.to_dbz = MagicMock(return_value=self.array)
        self.temporal.calc_map(self.array)
        self.array.lawr.to_dbz.assert_called_once()

    def test_create_cmap_returns_cluttermap(self):
        cmap = self.temporal.create_cluttermap(self.array)
        self.assertIsInstance(cmap, ClutterMap)
        self.assertEqual(len(cmap.layers), 1)
        self.assertEqual(list(cmap.layers.keys())[0],
                         'TemporalFilter')

    def test_create_cmap_calls_calc_map_if_array(self):
        self.temporal.calc_map(self.array)
        self.temporal.calc_map = MagicMock()
        _ = self.temporal.create_cluttermap(self.array)
        self.temporal.calc_map.assert_called_once_with(self.array)

    def test_create_cmap_sets_cmap_to_zero_if_no_fitted(self):
        cmap = self.temporal.create_cluttermap(self.array)
        np.testing.assert_equal(cmap.layers['TemporalFilter'], 0)

    def test_create_cmap_sets_cmap_to_zero_where_hist_map(self):
        self.array[..., 100, 100] = 10
        for _ in range(5):
            self.temporal.fit(self.array)
        cmap = self.temporal.create_cluttermap(self.array)
        np.testing.assert_equal(cmap.layers['TemporalFilter'], 0)

    def test_create_cmap_sets_clutter_to_one(self):
        self.temporal.fit(self.array)
        self.array[..., 100, 100] = 10
        self.temporal.fit(self.array)
        self.temporal.fit(self.array)
        cmap = self.temporal.create_cluttermap(self.array)
        np.testing.assert_allclose(cmap.layers['TemporalFilter'], 0, atol=1)
        self.assertEqual(cmap.layers['TemporalFilter'][..., 100, 100], 1)


if __name__ == '__main__':
    unittest.main()
