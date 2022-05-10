#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from copy import deepcopy
import time

# External modules
import cartopy.crs as ccrs
import numpy as np
from matplotlib import cm

# Internal modules
from pylawr.grid import PolarGrid
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.plot.plotter import Plotter
from pylawr.plot.subplot import Subplot
from pylawr.functions.plot import create_default_plotter, plot_rain_rate, \
    plot_rain_clutter, plot_reflectivity
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestPlotFunctions(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()
        self.array = create_array(self.grid)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)
        self.plot_path = '/tmp/pylawr_testing_plot.png'
        self.cluttermap = ClutterMap(
            'test', np.ones_like(self.array.values)
        )

    def tearDown(self):
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)

    def test_create_default_plotter_returns_plotter(self):
        subplots = ['map', 'header', 'colorbar']
        returned_plotter = create_default_plotter()
        self.assertIsInstance(returned_plotter, Plotter)
        for n in subplots:
            self.assertIn(n, returned_plotter.subplots.keys())
            self.assertIsInstance(returned_plotter.subplots[n], Subplot)

    def test_create_default_plotter_sets_projection(self):
        returned_plotter = create_default_plotter()
        map_subplot = returned_plotter.subplots.get('map')
        self.assertIsInstance(map_subplot.projection, ccrs.RotatedPole)

    def test_create_default_plotter_sets_extent(self):
        right_extent = dict(
            lon_min=9,
            lon_max=10,
            lat_min=53,
            lat_max=55
        )
        returned_plotter = create_default_plotter()
        old_extent = deepcopy(returned_plotter.subplots.get('map').extent)
        with self.assertRaises(AssertionError):
            self.assertDictEqual(old_extent, right_extent)
        returned_plotter = create_default_plotter(grid_extent=right_extent)
        self.assertDictEqual(returned_plotter.subplots.get('map').extent,
                             right_extent)

    def test_plot_rain_rate_saves_figure(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_rain_rate(self.array, self.plot_path)
        self.assertTrue(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)

    def test_plot_rain_rate_uses_plotter(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        plotter = create_default_plotter()
        plotter.plot()
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_rain_rate(self.array, self.plot_path, plotter=plotter)
        self.assertTrue(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)

    def test_plot_reflectivity_saves_figure(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_reflectivity(self.array, self.plot_path)
        self.assertTrue(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)

    def test_plot_reflectivity_takes_none_add_circle(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_reflectivity(self.array, self.plot_path, add_circle=None)
        self.assertTrue(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)

    def test_plot_reflectivity_uses_cmap(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_reflectivity(self.array, self.plot_path,
                          cmap=cm.get_cmap('viridis'))
        self.assertTrue(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)

    def test_plot_reflectivity_uses_default_cmap(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_reflectivity(self.array, self.plot_path)
        self.assertTrue(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)            
            
    @unittest.expectedFailure        
    def test_plot_reflectivity_cannot_take_string_as_cmap(self):
        timestr = time.strftime("%Y%m%d%H%M%S")
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_reflectivity(self.array, self.plot_path, cmap='viridis')
        self.assertFalse(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)         

    def test_plot_rain_clutter_saves_figure(self):
        right_extent = dict(
            lon_min=9,
            lon_max=10,
            lat_min=53,
            lat_max=55
        )
        timestr = time.strftime("%Y%m%d%H%M%S")
        self.plot_path = '/tmp/pylawr_testing_plot' + timestr + '.png'
        self.assertFalse(os.path.isfile(self.plot_path))
        plot_rain_clutter(self.array, self.cluttermap,
                          self.plot_path, grid_extent=None)
        self.assertTrue(os.path.isfile(self.plot_path))
        os.remove(self.plot_path)
        plot_rain_clutter(self.array, self.cluttermap,
                          self.plot_path, grid_extent=right_extent)
        self.assertTrue(os.path.isfile(self.plot_path))
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)


if __name__ == '__main__':
    unittest.main()
