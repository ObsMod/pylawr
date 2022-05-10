#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch
from copy import deepcopy

# External modules
import matplotlib.figure as mpl_figure
import matplotlib.colorbar as mpl_colorbar

import xarray as xr
import numpy as np
import cartopy.crs as ccrs

# Internal modules
from pylawr.grid import PolarGrid
from pylawr.plot.layer.lawr_header import LawrHeaderLayer
from pylawr.plot.layer.radarfield import RadarFieldLayer
from pylawr.plot.layer.colorbar import ColorbarLayer
from pylawr.utilities.helpers import create_array


logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))


BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestColorbarLayer(unittest.TestCase):
    def setUp(self):
        grid = PolarGrid()
        array = create_array(grid)
        self.figure = mpl_figure.Figure()
        self.axes = self.figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.radar_field_layer = RadarFieldLayer(array, grid=grid)
        self.layer = ColorbarLayer(layer=self.radar_field_layer)
        self.radar_field_layer.plot(ax=self.axes)

    def test_layer_access_private_layer(self):
        self.assertEqual(self.layer.layer, self.layer._layer)
        self.layer._layer = 123
        self.assertEqual(self.layer.layer, 123)

    def test_layer_sets_private_layer(self):
        self.assertEqual(self.layer.layer, self.layer._layer)
        self.layer._layer = None
        self.assertIsNone(self.layer._layer)
        self.layer.layer = self.radar_field_layer
        self.assertIsNotNone(self.layer._layer)
        self.assertEqual(self.layer._layer, self.radar_field_layer)

    def test_layer_raises_type_error_if_no_radar_field_layer(self):
        with self.assertRaises(TypeError):
            self.layer.layer = LawrHeaderLayer()

    def test_plot_gets_axes_position_from_axes(self):
        box = self.axes.get_position()
        with patch('matplotlib.axes.Axes.get_position', return_value=box) as p:
            self.layer.plot(ax=self.axes)
            p.assert_called_once_with()

    def test_plot_creates_colorbar(self):
        right_colorbar = self.axes.figure.colorbar(
            ax=self.axes, mappable=self.radar_field_layer.plot_store
        )
        with patch('matplotlib.figure.Figure.colorbar',
                   return_value=right_colorbar) as p:
            self.layer.plot(ax=self.axes)
            p.assert_called_once_with(
                ax=self.axes, mappable=self.radar_field_layer.plot_store
            )

    def test_plot_sets_colorbar_attribute(self):
        self.assertIsNone(self.layer.colorbar)
        right_colorbar = self.axes.figure.colorbar(
            ax=self.axes, mappable=self.radar_field_layer.plot_store
        )
        with patch('matplotlib.figure.Figure.colorbar',
                   return_value=right_colorbar):
            self.layer.plot(ax=self.axes)
        self.assertIsInstance(self.layer.colorbar, mpl_colorbar.Colorbar)
        self.assertEqual(self.layer.colorbar, right_colorbar)

    def test_plot_uses_settings_as_colorbar_settings(self):
        self.layer.settings = dict(ticks=[0, 1, 2])
        self.layer.plot(ax=self.axes)
        np.testing.assert_equal(self.layer.colorbar.locator.tick_values(0, 0),
                                np.array([0, 1, 2]))
        self.layer.settings = dict(label='test')
        self.layer.plot(ax=self.axes)
        # self.assertIsNone(self.layer.colorbar.locator)
        self.assertEqual(self.layer.colorbar.ax.get_ylabel(), 'test')

    def test_plot_moves_colorbar_to_box_pos_pad(self):
        box = self.axes.get_position()
        h_pad = self.layer.h_pad * box.width
        v_pad = self.layer.v_pad * box.height
        right_position = (
            box.x0 + h_pad,
            box.y0 + v_pad,
            box.width - 2 * h_pad,
            box.height - 2 * v_pad
        )
        self.layer.plot(ax=self.axes)
        cb_position = self.layer.colorbar.ax.get_position(original=True).bounds
        np.testing.assert_array_almost_equal(cb_position, right_position)

    def test_plot_sets_right_zorder(self):
        self.layer.zorder = None
        self.layer.plot(ax=self.axes)
        self.layer.zorder = 10
        self.layer.plot(ax=self.axes)
        self.assertEqual(10, self.layer.colorbar.ax.zorder)
        self.layer.zorder = 999
        self.layer.plot(ax=self.axes)
        self.assertEqual(999, self.layer.colorbar.ax.zorder)

    def test_plot_moves_axes_back_to_origional_position(self):
        box = self.axes.get_position()
        self.layer.plot(ax=self.axes)
        curr_box = self.axes.get_position()
        np.testing.assert_almost_equal(curr_box.bounds, box.bounds)

    def test_plot_does_not_change_zorder_of_parent_axes(self):
        ax_zorder = deepcopy(self.axes.zorder)
        self.layer.zorder = 10
        self.layer.plot(ax=self.axes)
        self.assertEqual(ax_zorder, self.axes.zorder)
        self.axes.zorder = 99
        self.layer.plot(ax=self.axes)
        self.assertEqual(99, self.axes.zorder)

    def test_plot_adds_colorbar_to_collection(self):
        self.assertFalse(self.layer.collection)
        self.layer.plot(ax=self.axes)
        self.assertListEqual([self.layer.colorbar], self.layer.collection)
        self.layer.plot(ax=self.axes)
        self.assertEqual(2, len(self.layer.collection))


if __name__ == '__main__':
    unittest.main()
