#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import logging
import os

# External modules
import numpy as np
import unittest

# Internal modules
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.filter.ring import RINGFilter
from pylawr.utilities.helpers import create_array
from pylawr.grid.polar import PolarGrid

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class RINGFilterTest(unittest.TestCase):
    def setUp(self):
        self.array = create_array(PolarGrid())

        self.f = RINGFilter()

    def test_create_cluttermap_with_addname(self):
        addname = '_test'
        name = str(self.f.__class__.__name__) + addname
        cluttermap = self.f.create_cluttermap(array=self.array, addname=addname)
        self.assertIn(name, cluttermap.layers.keys())

    def test_return_cluttermap(self):
        self.assertIsInstance(self.f.create_cluttermap(self.array),
                              ClutterMap)

    def test_removes_ring(self):
        self.array.values[0][:, 2] = 10.

        cmap = self.f.create_cluttermap(self.array)
        transformed_array = cmap.transform(self.array)
        self.assertEqual(np.count_nonzero(np.isnan(transformed_array.values)),
                         np.count_nonzero(self.array.values >
                                          self.f._threshold))

    def test_removes_short_ring(self):
        self.array.values[0][50:100, 0:1] = 10.
        self.array.values[0][70:80, 0:1] = 20.

        cmap = self.f.create_cluttermap(self.array)
        transformed_array = cmap.transform(self.array)

        self.assertEqual(np.count_nonzero(transformed_array > 0), 0)

    def test_holds_threshold_for_short_ring(self):
        self.array.values[0][5:10, 0] = 10.

        cmap = self.f.create_cluttermap(self.array)
        transformed_array = cmap.transform(self.array)

        self.assertEqual(np.count_nonzero(np.isnan(transformed_array)), 0)

    def test_calc_map(self):
        self.array.values[0][70:(71+self.f._wz), 0:1] = 20.
        self.f.calc_map(self.array)

        self.assertEqual(np.max(self.f.map),
                         self.f._wz)

    def test_remove_broad_ring(self):
        self.array.values[0][:, 0:2] = 10.

        cmap = self.f.create_cluttermap(self.array)
        transformed_array = cmap.transform(self.array)

        ring_variant = RINGFilter(ring_width=2)
        cmap_variant = ring_variant.create_cluttermap(self.array)
        transformed_variant = cmap_variant.transform(self.array)

        self.assertNotEqual(np.count_nonzero(transformed_array > 0), 0)
        self.assertEqual(np.count_nonzero(transformed_variant > 0), 0)


if __name__ == '__main__':
    unittest.main()
