#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 3/12/18
#
# Created for pattern
#
#
#    Copyright (C) {2018}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# system modules
import unittest
from unittest import mock
import logging
import os

# External modules
import xarray as xr

# Internal modules
from pylawr.field import get_verified_grid
from pylawr.transform.transformer import Transformer
from pylawr.grid import PolarGrid
from pylawr.utilities.helpers import create_array

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class SubTransformer(Transformer):
    def transform(self, array, grid=None, *args, **kwargs):
        return array


class TransformerTest(unittest.TestCase):
    # test environment transferred from old filter class
    def setUp(self):
        self.tformer = SubTransformer()
        self.grid = PolarGrid()
        self.array = create_array(self.grid)

    def test_verify_grid_uses_check_grid_of_radar_field(self):
        with mock.patch.object(self.array.lawr, 'check_grid',
                               return_value=self.grid) as p:
            get_verified_grid(self.array, self.grid)
            p.assert_called_once_with(grid=self.grid)

    def test_verify_grid_returns_grid_if_valid_and_given(self):
        verified_grid = get_verified_grid(self.array, self.grid)
        self.assertEqual(id(verified_grid), id(self.grid))

    def test_verify_grid_gets_grid_from_array_if_not_given(self):
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        verified_grid = get_verified_grid(self.array, None)
        self.assertEqual(id(verified_grid), id(self.grid))

    def test_verify_grid_raises_attribute_error_if_no_grid_possible(self):
        error_msg = 'A grid is needed, but no Grid is specified and no grid ' \
                    'is set for the array!'
        self.array.lawr._grid = None
        with self.assertRaises(AttributeError) as e:
            _ = get_verified_grid(self.array, None)
        self.assertEqual(str(e.exception), error_msg)


if __name__ == '__main__':
    unittest.main()
