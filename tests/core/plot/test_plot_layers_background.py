#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch

# External modules
import matplotlib.figure as mpl_figure

import cartopy.crs as ccrs
import cartopy.mpl as cmpl

# Internal modules
from pylawr.plot.layer.background import BackgroundLayer


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestBackgroundLayer(unittest.TestCase):
    def setUp(self):
        self.layer = BackgroundLayer()
        self.layer.resolution = 1
        self.figure = mpl_figure.Figure()
        self.axes = self.figure.add_subplot(111, projection=ccrs.PlateCarree())

    @patch('cartopy.mpl.geoaxes.GeoAxes.add_image', return_value=1)
    def test_plot_adds_img_tile_to_axes(self, p):
        self.layer.plot(ax=self.axes)
        p.assert_called_once_with(self.layer.img_tile, 1, zorder=0,
                                  **self.layer.settings)

    @patch('cartopy.mpl.geoaxes.GeoAxes.add_image', return_value=1)
    def test_plot_uses_set_settings_for_add_image(self, p):
        self.layer.plot(ax=self.axes)
        p.assert_called_once_with(self.layer.img_tile, 1, zorder=0)

    def test_plot_adds_returned_image_to_collection(self):
        self.assertFalse(self.layer.collection)
        self.layer.plot(ax=self.axes)
        self.assertListEqual(self.layer.collection, self.axes.get_images())

    @patch('cartopy.mpl.geoaxes.GeoAxes.add_image', return_value=1)
    def test_plot_sets_zorder_of_image(self, p):
        self.layer.settings = {}
        self.layer.zorder = 99
        self.layer.plot(ax=self.axes)
        p.assert_called_once_with(self.layer.img_tile, 1, zorder=99)

    @patch('cartopy.mpl.geoaxes.GeoAxes.add_image', return_value=1)
    def test_plot_sets_resolution_of_image(self, p):
        self.layer.settings = {}
        self.layer.resolution = 2
        self.layer.plot(ax=self.axes)
        p.assert_called_once_with(self.layer.img_tile, 2, zorder=0)


if __name__ == '__main__':
    unittest.main()
