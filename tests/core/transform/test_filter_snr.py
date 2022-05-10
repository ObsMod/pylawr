#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import logging
import unittest
import os

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.grid.polar import PolarGrid
from pylawr.transform.filter.snr import SNR
from pylawr.transform.temporal.noiseremover import NoiseRemover
from pylawr.transform.spatial.beamexpansion import BeamExpansion, \
    TAG_BEAM_EXPANSION_CORR
from pylawr.field import tag_array
from pylawr.functions.input import read_lawr_ascii

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class ClutterMapTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(DATA_PATH, 'lawr_data.txt')) as fh:
            self.array, _ = read_lawr_ascii(fh)

        # add tag to indicate that beam expansion has been corrected
        tag_array(self.array, TAG_BEAM_EXPANSION_CORR)

        self.noise_filter = NoiseRemover()

        self.beam_expansion_filter = BeamExpansion()

        # invert beam expansion effect
        self.array = self.beam_expansion_filter.transform(
            self.array.lawr.to_z())

        # fit the noise filter
        self.noise_filter.fit(self.array)

        # don't apply the noise filter, otherwise there are NaN values
        # self.array = self.noise_filter.transform(self.array)

        # Correct a second time the beam expansion
        self.array = self.beam_expansion_filter.transform(
            self.array.lawr.to_z())

        self.snr = SNR()

    def test_calc_map(self):
        self.snr.calc_map(self.array, self.noise_filter)

        test_array = self.beam_expansion_filter.transform(
            self.array.lawr.to_z())

        snr_array = test_array.values / self.noise_filter.noiselevel

        np.testing.assert_array_almost_equal(self.snr.map, snr_array)

    def test_create_cluttermap(self):
        cluttermap = self.snr.create_cluttermap(self.array, self.noise_filter)

        test_array = self.beam_expansion_filter.transform(
            self.array.lawr.to_z())

        snr_array = test_array.values / self.noise_filter.noiselevel

        snr_cmap = np.zeros(snr_array.shape)

        snr_cmap[snr_array > 0] = 1

        self.assertEqual(np.count_nonzero(cluttermap.layers['SNR'] == 1),
                         np.count_nonzero(snr_cmap == 1))


if __name__ == '__main__':
    unittest.main()
