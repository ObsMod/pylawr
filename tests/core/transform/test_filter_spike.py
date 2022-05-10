#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import logging
import os

# External modules
import xarray as xr
import numpy as np
import unittest

# Internal modules
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.filter.spike import SPKFilter
from pylawr.utilities.helpers import create_array
from pylawr.grid.polar import PolarGrid

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class SPKFilterTest(unittest.TestCase):
    def setUp(self):
        self.array = create_array(PolarGrid())

        self.spike = SPKFilter()

    def test_create_cluttermap_with_addname(self):
        addname = '_test'
        name = str(self.spike.__class__.__name__) + addname
        cluttermap = self.spike.create_cluttermap(array=self.array,
                                                  addname=addname)
        self.assertIn(name, cluttermap.layers.keys())

    def test_return_cluttermap(self):
        self.assertIsInstance(self.spike.create_cluttermap(self.array),
                              ClutterMap)

    def test_removes_spike(self):
        self.array.values[0][0:1, :] = 10.

        spike_cmap = self.spike.create_cluttermap(self.array)
        transformed_array = spike_cmap.transform(self.array)
        self.assertEqual(np.count_nonzero(np.isnan(transformed_array.values)),
                         np.count_nonzero(self.array.values >
                                          self.spike._threshold))

    def test_removes_short_spike(self):
        self.array.values[0][0:1, 50:100] = 10.
        self.array.values[0][0:1, 70:80] = 20.

        spike_cmap = self.spike.create_cluttermap(self.array)
        transformed_array = spike_cmap.transform(self.array)

        self.assertEqual(np.count_nonzero(transformed_array > 0), 0)

    def test_holds_threshold_for_short_spike(self):
        self.array.values[0][0, 5:8] = 10.

        spike_cmap = self.spike.create_cluttermap(self.array)
        transformed_array = spike_cmap.transform(self.array)

        self.assertEqual(np.count_nonzero(np.isnan(transformed_array)), 0)

    def test_calc_map(self):
        self.array.values[0][0:1, 70:(71+self.spike._wz)] = 20.
        self.spike.calc_map(self.array)

        self.assertEqual(np.max(self.spike.map),
                         self.spike._wz)

    def test_remove_broad_spike(self):
        self.array.values[0][0:2, :] = 10.

        spike_cmap = self.spike.create_cluttermap(self.array)
        transformed_array = spike_cmap.transform(self.array)

        spike_variant = SPKFilter(spike_width=2)
        spike_cmap_variant = spike_variant.create_cluttermap(self.array)
        transformed_variant = spike_cmap_variant.transform(self.array)

        self.assertNotEqual(np.count_nonzero(transformed_array > 0), 0)
        self.assertEqual(np.count_nonzero(transformed_variant > 0), 0)


if __name__ == '__main__':
    unittest.main()
