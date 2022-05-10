#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import logging
from unittest.mock import patch
import os
import unittest

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.field import tag_array
from pylawr.grid import PolarGrid
from pylawr.transform.spatial.beamexpansion import BeamExpansion,\
    TAG_BEAM_EXPANSION_CORR, TAG_BEAM_EXPANSION_UNCORR
from pylawr.utilities.helpers import create_array

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class BeamExpansionTest(unittest.TestCase):
    def setUp(self):
        self.f = BeamExpansion()
        self.grid = PolarGrid()
        self.refl = create_array(self.grid, 20., tag=None)

    def test_transform(self):
        transformed_array = self.f.transform(
            self.refl, grid=self.grid, inverse=False
        )
        right_array = self.refl * self.grid.center_distance ** 2
        xr.testing.assert_equal(
            right_array, transformed_array
        )
        self.assertEqual(transformed_array.attrs['tags'],
                         TAG_BEAM_EXPANSION_CORR)

    def test_transform_inverse(self):
        transformed_array = self.f.transform(
            self.refl, grid=self.grid, inverse=True
        )
        right_array = self.refl / self.grid.center_distance ** 2
        xr.testing.assert_equal(
            right_array, transformed_array
        )
        self.assertEqual(transformed_array.attrs['tags'],
                         TAG_BEAM_EXPANSION_UNCORR)

    def test_transform_inverse_consistency(self):
        transformed_array = self.f.transform(
                self.f.transform(self.refl, grid=self.grid, inverse=False),
                grid=self.grid,
                inverse=True
        )
        self.refl.lawr.add_tag(TAG_BEAM_EXPANSION_UNCORR)
        right_array = self.refl
        np.testing.assert_almost_equal(
            right_array.values, transformed_array.values, decimal=4
        )
        self.assertEqual(transformed_array.attrs['tags'],
                         TAG_BEAM_EXPANSION_UNCORR)

    def test_transform_uses_verify_grid(self):
        with patch('pylawr.transform.spatial.beamexpansion.get_verified_grid',
                   return_value=self.grid) as p:
            self.f.transform(self.refl, grid=self.grid)
            p.assert_called_once_with(self.refl, grid=self.grid)

    def test_transform_keeps_grid(self):
        self.refl = self.refl.lawr.set_grid_coordinates(self.grid)
        transformed_array = self.f.transform(
                self.refl,
                grid=self.grid
        )
        self.assertTrue(transformed_array.lawr.grid ==
                        self.refl.lawr.grid)

    def test_correction_keeps_attributes(self):
        tag_array(self.refl, 'test-tag')
        self.refl = self.refl.lawr.set_grid_coordinates(self.grid)
        transformed_array = self.f.transform(
            self.refl,
            grid=self.grid
        )
        self.assertIn('test-tag', transformed_array.attrs["tags"])


if __name__ == '__main__':
    unittest.main()
