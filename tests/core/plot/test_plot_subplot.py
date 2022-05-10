#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch
from copy import deepcopy

# External modules
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

import matplotlib.axes._subplots as mpl_subplots
import matplotlib.gridspec as mpl_gridspec
import matplotlib.figure as mpl_figure

import numpy as np

# Internal modules
from pylawr.datahandler import LawrHandler
from pylawr.grid import PolarGrid
from pylawr.plot.backend import BackendLoader
from pylawr.plot.plotter import Plotter
from pylawr.plot.layer import LawrHeaderLayer, RadarFieldLayer
from pylawr.plot.subplot import Subplot, default_tick_params


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestSubplot(unittest.TestCase):
    def setUp(self):
        self.subplot = Subplot()

    def test_ax_gets_private_axes(self):
        self.assertEqual(self.subplot.ax, self.subplot._ax)
        self.subplot._ax = 123
        self.assertEqual(self.subplot.ax, 123)

    def test_subplot_gets_ax_settings(self):
        self.assertDictEqual(self.subplot.ax_settings, dict())
        self.subplot = Subplot(get_frame_on=True)
        self.assertDictEqual(self.subplot.ax_settings, dict(get_frame_on=True))

    def test_subplot_layers_gets_private(self):
        layer = LawrHeaderLayer()
        self.assertEqual(self.subplot.layers, self.subplot._layers)
        self.subplot._layers = [layer]
        self.assertEqual(self.subplot.layers, self.subplot._layers)

    def test_add_layer_adds_layer_to_subplot_list(self):
        layer = LawrHeaderLayer()
        self.assertFalse(self.subplot.layers)
        self.subplot.add_layer(layer)
        self.assertListEqual(self.subplot.layers, [layer, ])

    def test_add_layer_raises_type_error_of_no_child_of_base_layer(self):
        with self.assertRaises(TypeError):
            self.subplot.add_layer(1234)

    def test_subplots_is_like_dict_with_ax_settings(self):
        self.subplot.ax_settings = {'test': 123}
        self.assertEqual(self.subplot['test'], 123)
        self.subplot['test1'] = 123
        self.assertDictEqual(self.subplot.ax_settings,
                             {'test': 123, 'test1': 123})
        self.subplot['test'] = 1
        self.assertDictEqual(self.subplot.ax_settings,
                             {'test': 1, 'test1': 123})
        del self.subplot['test1']
        self.assertDictEqual(self.subplot.ax_settings,
                             {'test': 1})
        self.subplot.update({'test1': 123})
        self.assertDictEqual(self.subplot.ax_settings,
                             {'test': 1, 'test1': 123})

    def test_layers_private_gets_private_list(self):
        self.assertEqual(self.subplot._layers, self.subplot._layers_list)
        self.subplot._layers_list = [123]
        self.assertEqual(self.subplot._layers, self.subplot._layers_list)

    def test_layers_private_sets_private_list(self):
        self.assertEqual(self.subplot._layers, self.subplot._layers_list)
        self.subplot._layers = [123]
        self.assertListEqual(self.subplot._layers_list, [123])

    def test_layers_private_none_sets_empty_list(self):
        self.subplot._layers = None
        self.assertListEqual(self.subplot._layers_list, [])

    def test_layers_private_converts_layers_list_to_list(self):
        self.subplot._layers = (1, 2, 3)
        self.assertListEqual(self.subplot._layers_list, [1, 2, 3])

    def test_layers_raises_type_error_if_not_iterable(self):
        with self.assertRaises(TypeError):
            self.subplot._layers = 'test'

    def test_projection_get_private_projection(self):
        projection = ccrs.PlateCarree()
        self.assertEqual(self.subplot.projection, self.subplot._projection)
        self.subplot._projection = projection
        self.assertEqual(self.subplot.projection, self.subplot._projection)

    def test_projection_sets_private_projection(self):
        projection = ccrs.PlateCarree()
        self.assertEqual(self.subplot.projection, self.subplot._projection)
        self.subplot.projection = projection
        self.assertEqual(id(self.subplot._projection), id(projection))

    def test_projection_raises_type_error_if_not_cartopy_or_none(self):
        projection = ccrs.PlateCarree()
        self.subplot.projection = projection
        self.subplot.projection = None
        self.assertIsNone(self.subplot._projection)
        with self.assertRaises(TypeError):
            self.subplot.projection = 123

    def test_new_axes_creates_new_axes(self):
        self.assertIsNone(self.subplot._ax)
        figure = mpl_figure.Figure()
        self.subplot.new_axes(figure, 111)
        self.assertIsInstance(self.subplot._ax, mpl_subplots.Axes)

    def test_new_axes_uses_projection(self):
        self.subplot.projection = ccrs.PlateCarree()
        figure = mpl_figure.Figure()
        self.subplot.new_axes(figure, 111)
        self.assertIsInstance(self.subplot._ax, GeoAxes)
        self.assertEqual(id(self.subplot._ax.projection),
                         id(self.subplot.projection))

    def test_plot_calls_new_axes(self):
        figure = mpl_figure.Figure()
        self.subplot.new_axes(figure, 111)
        with patch('pylawr.plot.subplot.Subplot.new_axes') as p:
            self.subplot.plot(figure, 111)
            p.assert_called_once_with(fig=figure, spec=111)

    def test_plot_calls_plot_layer_for_every_layer(self):
        figure = mpl_figure.Figure()
        layer = LawrHeaderLayer()
        self.subplot.add_layer(layer)
        self.subplot.add_layer(layer)
        with patch('pylawr.plot.subplot.Subplot.plot_layer_on_ax') as p:
            self.subplot.plot(figure, 111)
        self.assertEqual(p.call_count, len(self.subplot.layers))
        for call in p.call_args_list:
            self.assertDictEqual(call[1], {'layer': layer})

    def test_plot_layer_on_ax_calls_plot_routine_of_layer(self):
        figure = mpl_figure.Figure()
        layer = LawrHeaderLayer()
        self.subplot.new_axes(figure, 111)
        with patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer.plot') as p:
            self.subplot.plot_layer_on_ax(layer=layer)
            p.assert_called_once_with(ax=self.subplot.ax)

    def test_plot_layer_raises_typeerror_if_no_valid_layer(self):
        figure = mpl_figure.Figure()
        layer = 123
        self.subplot.new_axes(figure, 111)
        with self.assertRaises(TypeError):
            self.subplot.plot_layer_on_ax(layer=layer)

    def test_plot_layer_raises_valueerror_if_not_plotted_yet(self):
        layer = LawrHeaderLayer()
        with self.assertRaises(ValueError):
            self.subplot.plot_layer_on_ax(layer=layer)

    def test_plotted_checks_if_axes_is_created(self):
        self.assertFalse(self.subplot.plotted)
        figure = mpl_figure.Figure()
        self.subplot.new_axes(figure, 111)
        self.assertTrue(self.subplot.plotted)

    def test_plotted_works_for_geoaxes(self):
        self.subplot.projection = ccrs.PlateCarree()
        figure = mpl_figure.Figure()
        self.assertFalse(self.subplot.plotted)
        self.subplot.new_axes(figure, 111)
        self.assertTrue(self.subplot.plotted)

    def test_getattr_uses_methods_from_ax(self):
        figure = mpl_figure.Figure()
        self.subplot.new_axes(figure, 111)
        with patch('matplotlib.axes._subplots.Axes.set_frame_on') as p:
            self.subplot.set_frame_on(False)
            p.assert_called_once_with(False)

    def test_getattr_raises_attribute_error_if_not_available(self):
        with self.assertRaises(AttributeError) as e:
            self.subplot.set_frame_on(False)

    def test_new_axes_uses_ax_settings(self):
        figure = mpl_figure.Figure()
        self.subplot.new_axes(figure, 111, frameon=False)
        self.assertFalse(self.subplot.ax.get_frame_on())

    def test_plot_calls_new_axes_with_ax_settings(self):
        figure = mpl_figure.Figure()
        self.subplot['frameon'] = False
        self.subplot.new_axes(figure, 111, frameon=False)
        with patch('pylawr.plot.subplot.Subplot.new_axes') as p:
            self.subplot.plot(fig=figure, spec=111)
            p.assert_called_once_with(fig=figure, spec=111, frameon=False)

    def test_new_axes_sets_default_tick_params(self):
        figure = mpl_figure.Figure()
        with patch('matplotlib.axes._subplots.Axes.tick_params') as p:
            self.subplot.new_axes(figure, 111)
            p.assert_called_with(**default_tick_params)

    def test_extent_settings_returns_extent_settings(self):
        self.assertDictEqual(self.subplot.extent_settings,
                             self.subplot._extent_settings)
        self.subplot._extent_settings = {'test': 123}
        self.assertDictEqual(self.subplot.extent_settings,
                             {'test': 123})

    def test_auto_extent_gets_auto_key_from_extent_settings(self):
        self.assertEqual(self.subplot.auto_extent,
                         self.subplot.extent_settings['auto'])
        self.subplot._extent_settings['auto'] = 123
        self.assertEqual(self.subplot.auto_extent, 123)

    def test_auto_extent_sets_auto_key_from_extent_settings(self):
        self.assertTrue(self.subplot.auto_extent)
        self.subplot.auto_extent = False
        self.assertFalse(self.subplot.auto_extent)
        self.assertFalse(self.subplot.extent_settings['auto'])

    def test_auto_extent_raises_type_error_if_no_boolean(self):
        with self.assertRaises(TypeError):
            self.subplot.auto_extent = 123
            self.subplot.auto_extent = dict()
            self.subplot.auto_extent = list()
            self.subplot.auto_extent = 'test'

    def test_extent_returns_lon_and_lat_keys_from_extent_settings(self):
        lon_lat_keys = ['lon_min', 'lon_max', 'lat_min', 'lat_max']
        lon_lat_from_settings = {k: self.subplot.extent_settings[k]
                                 for k in lon_lat_keys}
        self.assertDictEqual(self.subplot.extent, lon_lat_from_settings)
        self.subplot._extent_settings['lon_min'] = -128
        with self.assertRaises(AssertionError):
            self.assertDictEqual(self.subplot.extent, lon_lat_from_settings)
        lon_lat_from_settings = {k: self.subplot.extent_settings[k]
                                 for k in lon_lat_keys}
        self.assertDictEqual(self.subplot.extent, lon_lat_from_settings)

    def test_extent_sets_new_keys_from_dict_to_extent_settings(self):
        lon_lat_keys = ['lon_min', 'lon_max', 'lat_min', 'lat_max']
        lon_lat_dict = {k: 123 for k in lon_lat_keys}
        lon_lat_from_settings = {k: self.subplot.extent_settings[k]
                                 for k in lon_lat_keys}
        self.assertDictEqual(self.subplot.extent, lon_lat_from_settings)
        self.subplot.extent = lon_lat_dict
        with self.assertRaises(AssertionError):
            self.assertDictEqual(lon_lat_dict, lon_lat_from_settings)
        lon_lat_from_settings = {k: self.subplot.extent_settings[k]
                                 for k in lon_lat_keys}
        self.assertDictEqual(lon_lat_dict, lon_lat_from_settings)

    def test_extent_updates_extent_settings(self):
        lon_lat_keys = ['lon_min', 'lon_max', 'lat_min', 'lat_max']
        lon_lat_dict = {k: 123 for k in lon_lat_keys}
        extent_settings = deepcopy(self.subplot.extent_settings)
        self.assertDictEqual(self.subplot.extent_settings, extent_settings)
        extent_settings.update(lon_lat_dict)
        self.subplot.extent = lon_lat_dict
        self.assertDictEqual(self.subplot.extent_settings, extent_settings)

    def test_extent_filters_keys(self):
        lon_lat_keys = ['lon_min', 'lon_max', 'lat_min', 'lat_max']
        lon_lat_dict = {k: 123 for k in lon_lat_keys}
        update_dict = deepcopy(lon_lat_dict)
        update_dict['projection'] = 123
        extent_settings = deepcopy(self.subplot.extent_settings)
        self.assertDictEqual(self.subplot.extent_settings, extent_settings)
        extent_settings.update(lon_lat_dict)
        self.subplot.extent = update_dict
        self.assertDictEqual(self.subplot.extent_settings, extent_settings)

    def test_extent_raises_typeerror_if_no_dict(self):
        with self.assertRaises(TypeError):
            self.subplot.extent = 123

    def test_update_extent_raises_type_error_if_not_plotted(self):
        figure = mpl_figure.Figure()
        projection = ccrs.PlateCarree()
        gridspec = mpl_gridspec.GridSpec(14, 13)
        map_spec = gridspec[slice(12), slice(2, 15)]
        self.subplot.projection = projection
        self.subplot.new_axes(figure, map_spec)
        self.subplot.update_extent(map_spec)
        self.subplot._ax = None
        with self.assertRaises(TypeError):
            self.subplot.update_extent(map_spec)

    def test_update_extent_raises_type_error_if_no_geoaxes(self):
        figure = mpl_figure.Figure()
        projection = ccrs.PlateCarree()
        gridspec = mpl_gridspec.GridSpec(14, 13)
        map_spec = gridspec[slice(12), slice(2, 15)]
        self.subplot.new_axes(figure, map_spec)
        with self.assertRaises(TypeError):
            self.subplot.update_extent(map_spec)
        self.subplot.projection = projection
        self.subplot.new_axes(figure, map_spec)
        self.subplot.update_extent(map_spec)

    def test_update_extent_uses_extent_if_extent_set(self):
        figure = mpl_figure.Figure()
        self.subplot.projection = ccrs.PlateCarree()
        self.subplot.new_axes(figure, 111)
        extent = (9, 11, 53, 54)
        extent_dict = dict(zip(self.subplot._extent_keys, extent))
        self.subplot.auto_extent = False
        self.subplot.extent = extent_dict
        old_extent = self.subplot.ax.get_extent()
        self.assertTupleEqual(self.subplot.ax.get_extent(), old_extent)
        self.subplot.update_extent()
        with self.assertRaises(AssertionError):
            self.assertTupleEqual(self.subplot.ax.get_extent(), old_extent)
        self.assertTupleEqual(self.subplot.ax.get_extent(), extent)

    def test_update_extent_doesnt_update_extent_if_not_set(self):
        figure = mpl_figure.Figure()
        self.subplot.projection = ccrs.PlateCarree()
        self.subplot.new_axes(figure, 111)
        self.subplot.auto_extent = False
        for k in self.subplot._extent_keys:
            self.subplot._extent_settings.pop(k, None)
        old_extent = self.subplot.ax.get_extent()
        self.assertTupleEqual(self.subplot.ax.get_extent(), old_extent)
        self.subplot.update_extent()
        self.assertTupleEqual(self.subplot.ax.get_extent(), old_extent)

    @patch('pylawr.plot.subplot.Subplot._calc_auto_extent',
           return_value=(0, 11, 53, 54))
    def test_update_extent_calls_extend_auto_expand(self, p):
        figure = mpl_figure.Figure()
        self.subplot.projection = ccrs.PlateCarree()
        self.subplot.new_axes(figure, 111)
        self.subplot.auto_extent = True
        self.subplot.update_extent()
        p.assert_called_once()

    @patch('pylawr.plot.subplot.Subplot._calc_auto_extent',
           return_value=(0, 11, 53, 54))
    def test_update_extent_calls_not_auto_expand_if_not_auto_extent(self, p):
        figure = mpl_figure.Figure()
        self.subplot.projection = ccrs.PlateCarree()
        self.subplot.new_axes(figure, 111)
        self.subplot.auto_extent = False
        self.subplot.update_extent()
        p.assert_not_called()

    @patch('pylawr.plot.subplot.Subplot._calc_auto_extent',
           return_value=(0, 11, 53, 54))
    def test_update_extent_sets_auto_extent(self, p):
        figure = mpl_figure.Figure()
        self.subplot.projection = ccrs.PlateCarree()
        self.subplot.new_axes(figure, 111)
        for k in self.subplot._extent_keys:
            self.subplot._extent_settings.pop(k, None)
        self.subplot.auto_extent = True
        old_extent = self.subplot.ax.get_extent()
        self.assertTupleEqual(self.subplot.ax.get_extent(), old_extent)
        self.subplot.update_extent()
        with self.assertRaises(AssertionError):
            self.assertTupleEqual(self.subplot.ax.get_extent(), old_extent)
        self.assertTupleEqual(self.subplot.ax.get_extent(), (0, 11, 53, 54))

    def test_get_subplot_aspect_returns_right_aspect(self):
        figure = mpl_figure.Figure(figsize=(13, 9))
        gridspec = mpl_gridspec.GridSpec(14, 16)
        map_spec = gridspec[slice(5), slice(2, 15)]
        self.subplot.new_axes(figure, map_spec)

        slice_rows = 5
        slice_cols = 13
        slice_aspect = slice_cols / slice_rows
        grid_aspect = 16 / 14
        figure_aspect = figure.get_figwidth() / figure.get_figheight()

        right_aspect = slice_aspect / grid_aspect * figure_aspect
        returned_aspect = self.subplot._get_subplot_aspect(map_spec)

        self.assertEqual(right_aspect, returned_aspect)

    def test_calc_auto_extent_returns_enlarged_height_extent(self):
        figure = mpl_figure.Figure(figsize=(13, 9))
        gridspec = mpl_gridspec.GridSpec(14, 16)
        map_spec = gridspec[slice(12), slice(2, 15)]
        self.subplot.projection = ccrs.PlateCarree()
        self.subplot.new_axes(figure, map_spec)
        subplot_aspect = self.subplot._get_subplot_aspect(map_spec)
        extent = list(self.subplot._ax.get_extent())
        extent_width = extent[1]-extent[0]
        extent_height = extent[3]-extent[2]
        curr_aspect = extent_width/extent_height
        scaling_factor = curr_aspect / subplot_aspect
        returned_extent = self.subplot._calc_auto_extent(map_spec)
        returned_height = returned_extent[3] - returned_extent[2]
        height_scaling = returned_height / extent_height
        self.assertAlmostEqual(height_scaling, scaling_factor)

    def test_calc_auto_extent_returns_enlarged_width_extent(self):
        figure = mpl_figure.Figure(figsize=(13, 4))
        gridspec = mpl_gridspec.GridSpec(14, 13)
        map_spec = gridspec[slice(12), slice(2, 15)]
        self.subplot.projection = ccrs.PlateCarree()
        self.subplot.new_axes(figure, map_spec)
        subplot_aspect = self.subplot._get_subplot_aspect(map_spec)
        extent = list(self.subplot._ax.get_extent())
        extent_width = extent[1]-extent[0]
        extent_height = extent[3]-extent[2]
        curr_aspect = extent_width/extent_height
        scaling_factor = subplot_aspect / curr_aspect
        returned_extent = self.subplot._calc_auto_extent(map_spec)
        returned_width = returned_extent[1] - returned_extent[0]
        width_scaling = returned_width / extent_width
        self.assertAlmostEqual(width_scaling, scaling_factor)

    def test_plot_adjusts_extent(self):
        figure = mpl_figure.Figure(figsize=(13, 4))
        gridspec = mpl_gridspec.GridSpec(14, 13)
        map_spec = gridspec[slice(12), slice(2, 15)]
        self.subplot.projection = ccrs.PlateCarree()
        with patch('pylawr.plot.subplot.Subplot.update_extent') as p:
            self.subplot.plot(figure, map_spec)
            p.assert_called_once_with(spec=map_spec)

    @patch('pylawr.plot.subplot.Subplot.plot_layer_on_ax')
    def test_add_layers_calls_plot_layer_on_ax_if_plotted(self, p):
        layer = LawrHeaderLayer()
        figure = mpl_figure.Figure(figsize=(13, 4))
        gridspec = mpl_gridspec.GridSpec(14, 13)
        map_spec = gridspec[slice(12), slice(2, 15)]
        self.subplot.new_axes(figure, map_spec)
        self.subplot.add_layer(layer)
        p.assert_called_once_with(layer=layer)

    def test_swap_layer_raises_type_error_if_new_layer_not_valid(self):
        layer = LawrHeaderLayer()
        self.subplot.add_layer(layer)
        with self.assertRaises(TypeError):
            self.subplot.swap_layer(None, layer)
        self.subplot.swap_layer(layer, layer)

    def test_swap_layer_raises_type_error_if_old_layer_not_valid(self):
        layer = LawrHeaderLayer()
        with self.assertRaises(TypeError):
            self.subplot.swap_layer(layer, None)

    def test_swap_layer_raises_key_error_if_old_layer_not_in_layers(self):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        with self.assertRaises(KeyError):
            self.subplot.swap_layer(layer, layer_1)

    def test_swap_layer_calls_remove_from_layer(self):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        self.subplot.add_layer(layer)
        with patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer.remove') as p:
            self.subplot.swap_layer(layer_1, layer)
            p.assert_called_once_with()

    def test_swap_layer_adds_new_layer_to_the_position_of_old_layer(self):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        layer_2 = LawrHeaderLayer()
        self.subplot.add_layer(layer)
        self.subplot.add_layer(layer_1)
        self.subplot.swap_layer(layer_2, layer)
        self.assertListEqual(self.subplot.layers, [layer_2, layer_1])

    def test_swap_layer_calls_plot_from_layer_if_plotted(self):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        self.subplot.add_layer(layer)
        figure = mpl_figure.Figure(figsize=(13, 4))
        gridspec = mpl_gridspec.GridSpec(14, 13)
        map_spec = gridspec[slice(12), slice(2, 15)]
        self.subplot.plot(fig=figure, spec=map_spec)
        with patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer.plot') as p:
            self.subplot.swap_layer(layer_1, layer)
            p.assert_called_once_with(ax=self.subplot.ax)

    def test_swap_layer_copies_zorder_if_set(self):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        self.subplot.add_layer(layer)
        self.subplot.swap_layer(layer_1, layer)
        self.assertEqual(layer_1.zorder, 0)
        layer_1.zorder = 100
        self.subplot.swap_layer(layer, layer_1)
        self.assertEqual(layer.zorder, 100)


if __name__ == '__main__':
    unittest.main()
