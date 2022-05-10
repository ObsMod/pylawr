#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import logging
import os
import unittest

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.filter.spin import SPINFilter
from pylawr.functions.input import read_lawr_ascii

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class SPINFilterTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(DATA_PATH, 'lawr_data.txt')) as fh:
            self.array, _ = read_lawr_ascii(fh)

        self.spin = SPINFilter()

    def test_create_cluttermap_with_addname(self):
        addname = '_test'
        name = str(self.spin.__class__.__name__) + addname
        cluttermap = self.spin.create_cluttermap(array=self.array,
                                                 addname=addname)
        self.assertIn(name, cluttermap.layers.keys())

    def test_return_cluttermap(self):
        self.assertIsInstance(self.spin.create_cluttermap(self.array),
                              ClutterMap)

    def test_reshape_pad(self):
        wd = self.spin._wz

        nb_pad = int(wd / 2) + 1

        tarray = np.pad(self.array,
                        ((0, 0),
                         (0, 0),
                         (nb_pad, nb_pad)
                         ), 'reflect')

        dist = tarray.shape[2]

        tarray = tarray[:, :, (nb_pad):(dist - nb_pad)]

        np.testing.assert_array_equal(tarray, self.array)

    def test_calc_map(self):
        wd = self.spin._wz
        th = self.spin._threshold

        tarray = np.pad(self.array,
                        ((0, 0),
                         (0, 0),
                         (1, 1)),
                        'reflect')

        grad_left = np.subtract(tarray,
                                np.roll(tarray, 1, 2))
        grad_right = np.subtract(np.roll(tarray, -1, 2),
                                 tarray)

        grad_left = grad_left[:, :, 1:(tarray.shape[2] - 1)]
        grad_right = grad_right[:, :, 1:(tarray.shape[2] - 1)]

        grad_left = np.pad(grad_left,
                           ((0, 0),
                            (0, 0),
                            (int(wd / 2) + 1, int(wd / 2) + 1)),
                           'reflect')
        grad_right = np.pad(grad_right,
                            ((0, 0),
                             (0, 0),
                             (int(wd / 2) + 1, int(wd / 2) + 1)),
                            'reflect')

        # first condition
        map_sign = np.multiply(np.sign(grad_left), -1. * np.sign(grad_right))
        map_sign[map_sign == -1] = 0
        map_sign[np.isnan(map_sign)] = 0

        # second condition
        map_mean = np.add(abs(grad_left),
                          abs(grad_right)) / 2.
        map_mean[np.isnan(map_mean)] = 0
        map_mean[map_mean <= th] = 0
        map_mean[map_mean > th] = 1

        # combination
        map = np.zeros(map_mean.shape)

        for cgate in range(int(-wd / 2), int(wd / 2) + 1):
            map += np.multiply(np.roll(map_sign, cgate, 2),
                               np.roll(map_mean, cgate, 2))
        map = map / wd
        map = map[:, :, (int(wd / 2) + 1):(map.shape[2] - int(wd / 2) - 1)]

        self.spin.calc_map(self.array)

        np.testing.assert_array_almost_equal(map, self.spin.map, decimal=3)

    def test_create_cluttermap(self):
        wd = self.spin._wz
        wc = self.spin._wc
        filter = 'SPINFilter'

        spin_cmap = self.spin.create_cluttermap(self.array)

        map = self.spin.map

        clutter = np.zeros(map.shape)
        clutter[map > wc] = 1

        cmap_layer = {filter: clutter}

        np.testing.assert_array_equal(cmap_layer[filter],
                                      spin_cmap.layers[filter])


if __name__ == '__main__':
    unittest.main()
