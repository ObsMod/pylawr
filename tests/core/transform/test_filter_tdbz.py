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
from pylawr.transform.filter.tdbz import TDBZFilter
from pylawr.functions.input import read_lawr_ascii

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TDBZFilterTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(DATA_PATH, 'lawr_data.txt')) as fh:
            self.array, _ = read_lawr_ascii(fh)

        self.tdbz = TDBZFilter()

    def test_create_cluttermap_with_addname(self):
        addname = '_test'
        name = str(self.tdbz.__class__.__name__) + addname
        cluttermap = self.tdbz.create_cluttermap(array=self.array,
                                                 addname=addname)
        self.assertIn(name, cluttermap.layers.keys())

    def test_return_cluttermap(self):
        self.assertIsInstance(self.tdbz.create_cluttermap(self.array),
                              ClutterMap)

    def test_reshape_pad(self):
        wd = self.tdbz._wz

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
        wd = self.tdbz._wz
        map = 0

        tarray = np.pad(self.array,
                        ((0, 0),
                         (0, 0),
                         (int(wd / 2)+1, int(wd / 2)+1)),
                        'reflect')

        diffsq = np.zeros(tarray.shape + (wd,))

        for cgate in range(int(-wd / 2), int(wd / 2) + 1):
            diffsq[:, :, :, cgate] = np.square(
                np.subtract(np.roll(tarray, cgate, 2),
                            np.roll(tarray, (cgate - 1), 2))
                )

        map = np.nanmean(diffsq, 3)

        map = map[:, :, (int(wd / 2)+1):(tarray.shape[2]-int(wd / 2)-1)]

        self.tdbz.calc_map(self.array)

        np.testing.assert_array_almost_equal(map, self.tdbz.map, decimal=3)

    def test_create_cluttermap(self):
        filter = 'TDBZFilter'

        tdbz_cmap = self.tdbz.create_cluttermap(self.array)

        map = self.tdbz.map

        clutter = np.zeros(map.shape)
        clutter[map >= self.tdbz._threshold] = 1

        cmap_layer = {filter: clutter}

        np.testing.assert_array_equal(cmap_layer[filter],
                                      tdbz_cmap.layers[filter])


if __name__ == '__main__':
    unittest.main()
