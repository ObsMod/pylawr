#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import unittest
import os
from unittest.mock import patch

# External modules
import numpy as np
import matplotlib.figure as mpl_figure
import matplotlib.gridspec as mpl_gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Internal modules
from pylawr.plot import NotPlottedError
from pylawr.plot.backend import BackendLoader
from pylawr.plot.plotter import Plotter, default_grid_slices
from pylawr.plot.subplot import Subplot
from pylawr.plot.layer import LawrHeaderLayer

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

RandomState = np.random.RandomState(42)


class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.plotter = Plotter('agg')

    def test_plotter_acts_like_a_dict_for_fig_settings(self):
        self.plotter.fig_settings = {'test': 123}
        self.assertEqual(self.plotter['test'], 123)
        self.plotter['test1'] = 123
        self.assertDictEqual(self.plotter.fig_settings,
                             {'test': 123, 'test1': 123})
        self.plotter['test'] = 1
        self.assertDictEqual(self.plotter.fig_settings,
                             {'test': 1, 'test1': 123})
        del self.plotter['test1']
        self.assertDictEqual(self.plotter.fig_settings,
                             {'test': 1})
        self.plotter.update({'test1': 123})
        self.assertDictEqual(self.plotter.fig_settings,
                             {'test': 1, 'test1': 123})

    def test_backend_name_creates_backend_loader_with_same_name(self):
        self.plotter = Plotter(backend_name='pdf')
        self.assertIsInstance(self.plotter._backend, BackendLoader)
        self.assertEqual(self.plotter._backend.name, 'pdf')

    def test_backend_name_gets_backend_name(self):
        self.plotter = Plotter(backend_name='pdf')
        self.plotter._backend._name = 'test'
        self.assertEqual(self.plotter.backend_name, 'test')

    def test_backend_name_setter_sets_backend(self):
        self.plotter = Plotter(backend_name='agg')
        self.plotter.backend_name = 'pdf'
        self.assertEqual(self.plotter._backend._name, 'pdf')

    def test_new_figure_sets_figure(self):
        self.plotter._figure = None
        self.assertIsNone(self.plotter._figure)
        self.plotter.new_figure()
        self.assertIsNotNone(self.plotter._figure)
        self.assertIsInstance(self.plotter._figure, mpl_figure.Figure)

    def test_new_figure_sets_canvas(self):
        self.plotter._canvas = None
        self.assertIsNone(self.plotter._canvas)
        self.plotter.new_figure()
        self.assertIsNotNone(self.plotter._canvas)
        self.assertIsInstance(self.plotter._canvas, FigureCanvasAgg)
        self.assertEqual(id(self.plotter._canvas.figure),
                         id(self.plotter._figure))

    def test_new_figure_uses_fig_settings(self):
        figsize = (12, 10)
        with self.assertRaises(AssertionError):
            self.plotter.new_figure()
            self.assertTupleEqual(
                figsize, tuple(self.plotter._figure.get_size_inches())
            )
        self.plotter.new_figure(figsize=figsize)
        self.assertTupleEqual(
            figsize, tuple(self.plotter._figure.get_size_inches())
        )

    def test_figure_property_returns_private_figure(self):
        self.assertEqual(self.plotter.figure, self.plotter._figure)
        self.plotter._figure = None
        self.assertIsNone(self.plotter.figure)

    def test_grid_size_gets_grid_size(self):
        self.assertTupleEqual(self.plotter.grid_size, self.plotter._grid_size)
        self.plotter._grid_size = None
        self.assertIsNone(self.plotter.grid_size)

    def test_grid_size_setter_sets_grid_size(self):
        self.assertTupleEqual(self.plotter.grid_size, self.plotter._grid_size)
        self.plotter.grid_size = 16
        self.assertTupleEqual(self.plotter._grid_size, (16, 16))

    def test_gridspec_returns_grid_with_grid_size(self):
        self.assertIsInstance(self.plotter.gridspec, mpl_gridspec.GridSpec)
        self.assertTupleEqual(self.plotter.grid_size,
                              self.plotter.gridspec.get_geometry())
        self.plotter.grid_size = (12, 8)
        self.assertTupleEqual(self.plotter.grid_size,
                              self.plotter.gridspec.get_geometry())

    def test_grid_slices_gets_grid_slices_if_not_none(self):
        self.plotter._grid_slices = {'header': slice(None, None)}
        self.assertIsNotNone(self.plotter.grid_slices)
        self.assertDictEqual(self.plotter._grid_slices,
                             self.plotter.grid_slices)

    def test_grid_slices_setter_sets_grid_slices(self):
        slices = {'header': (slice(None, None), slice(None, None))}
        self.plotter._grid_slices = {}
        self.plotter._add_grid_slices_to_subplot(slices)
        self.assertDictEqual(self.plotter._grid_slices, slices)

    def test_grid_slices_setter_sets_default_grid_slices_if_none(self):
        self.plotter._subplots = {}
        self.plotter._grid_slices = {}
        self.plotter._add_grid_slices_to_subplot(None)
        self.assertIsNotNone(self.plotter.grid_slices)
        self.assertDictEqual(default_grid_slices, self.plotter.grid_slices)

    def test_grid_slices_to_subplot_raises_type_error_if_no_dict(self):
        with self.assertRaises(TypeError):
            self.plotter._add_grid_slices_to_subplot(grid_slices=123)

    def test_grid_slices_adds_subplots_for_new_slices(self):
        self.plotter._subplots = {}
        self.plotter._add_grid_slices_to_subplot({
            'header': (slice(None, None), slice(None, None))
        })
        self.assertIn('header', self.plotter.subplots.keys())
        self.assertIsInstance(self.plotter.subplots['header'], Subplot)

    def test_add_subplot_raises_key_error_if_already_in_subplots(self):
        self.plotter._subplots = {'test': 1}
        with self.assertRaises(KeyError):
            self.plotter.add_subplot(name='test')

    def test_add_subplot_adds_subplot_to_gridslices(self):
        self.plotter._grid_slices = {}
        self.assertFalse(self.plotter.grid_slices)
        self.plotter.add_subplot(name='test')
        self.assertDictEqual(self.plotter.grid_slices,
                             {'test': (slice(None, None), slice(None, None))})

    def test_add_subplot_adds_subplot_to_subplots(self):
        self.plotter._subplots = {}
        self.assertFalse(self.plotter.subplots)
        self.plotter.add_subplot(name='test')
        self.assertIn('test', self.plotter.subplots.keys())
        self.assertIsInstance(self.plotter.subplots['test'], Subplot)

    def test_add_subplot_uses_ax_settings(self):
        self.plotter._subplots = {}
        self.assertFalse(self.plotter.subplots)
        self.plotter.add_subplot(name='test', set_frame_on=True)
        self.assertDictEqual(self.plotter.subplots['test'].ax_settings,
                             dict(set_frame_on=True))

    def test_subplots_returns_subplot(self):
        self.assertEqual(self.plotter.subplots, self.plotter._subplots)
        self.plotter._subplots = 1234
        self.assertEqual(self.plotter.subplots, self.plotter._subplots)

    def test_del_subplot_deletes_subplot_from_subplots_dict(self):
        self.assertTrue(self.plotter.subplots)
        self.assertIn('map', self.plotter.subplots.keys())
        self.plotter.del_subplot('map')
        self.assertFalse(self.plotter.subplots)
        self.assertNotIn('map', self.plotter.subplots.keys())

    def test_del_subplot_deletes_grid_slice_from_grid_slices_dict(self):
        self.assertTrue(self.plotter.grid_slices)
        self.assertIn('map', self.plotter.grid_slices.keys())
        self.plotter.del_subplot('map')
        self.assertNotIn('map', self.plotter.grid_slices.keys())
        self.assertFalse(self.plotter.grid_slices)

    def test_change_grid_slice_raises_keyerror_if_name_not_available(self):
        with self.assertRaises(KeyError):
            self.plotter.change_grid_slice('test', (slice(None), slice(None)))

    def test_change_grid_slice_changes_grid_slice(self):
        new_grid_slice = (slice(0, 1), slice(None, None))
        with self.assertRaises(AssertionError):
            self.assertTupleEqual(self.plotter.grid_slices['map'],
                                  new_grid_slice)
        self.plotter.change_grid_slice('map', new_grid_slice)
        self.assertEqual(self.plotter.grid_slices['map'], new_grid_slice)

    def test_plotted_returns_if_figure_is_created(self):
        self.plotter._figure = None
        self.assertFalse(self.plotter.plotted)
        self.plotter.new_figure()
        self.assertTrue(self.plotter.plotted)

    def test_add_layer_checks_if_subplot_exists(self):
        layer = LawrHeaderLayer()
        err_msg = 'The given subplot {0:s} does not exist yet, available ' \
                  'subplots are: {1:s}'.format(
                      'header', ','.join(self.plotter.subplots.keys())
                  )
        with self.assertRaises(KeyError, msg=err_msg):
            self.plotter.add_layer('header', layer)

    def test_add_layer_adds_layer_to_given_subplot(self):
        layer = LawrHeaderLayer()
        self.assertFalse(self.plotter.subplots['map'].layers)
        self.plotter.add_layer('map', layer)
        self.assertEqual(id(self.plotter.subplots['map'].layers[0]),
                         id(layer))

    def test_plot_calls_new_figure(self):
        self.plotter.new_figure()
        with patch('pylawr.plot.plotter.Plotter.new_figure') as p:
            self.plotter.plot()
            p.assert_called_once()

    def test_plot_calls_new_figure_with_fig_settings(self):
        self.plotter['figsize'] = (12, 9)
        self.plotter.new_figure(**self.plotter.fig_settings)
        with patch('pylawr.plot.plotter.Plotter.new_figure') as p:
            self.plotter.plot()
            p.assert_called_once_with(figsize=(12, 9))

    def test_plot_calls_plot_of_every_subplot(self):
        self.plotter.add_subplot('test')
        with patch('pylawr.plot.subplot.Subplot.plot') as p:
            self.plotter.plot()
            p.assert_called()
        self.assertEqual(p.call_count, len(self.plotter.subplots))
        for call in p.call_args_list:
            kwargs = call[1]
            self.assertListEqual(list(kwargs.keys()), ['fig', 'spec'])

    def test_figure_methods_are_available(self):
        self.plotter.new_figure()
        with patch('matplotlib.figure.Figure.savefig') as p:
            self.plotter.savefig('test.png')
            p.assert_called_once_with('test.png')

    def test_figure_method_raises_attribute_error_if_not_available(self):
        with self.assertRaises(AttributeError):
            self.plotter.savefig('test.png')

    def test_gridspec_uses_gridspec_settings(self):
        gridspec_before = self.plotter.gridspec
        self.plotter.gridspec_settings['hspace'] = 0.1
        gridspec_after = self.plotter.gridspec
        self.assertNotEqual(gridspec_before.hspace, gridspec_after.hspace)

    def test_remove_subplot_calls_axes_remove_from_figure_if_plotted(self):
        self.plotter.plot()
        self.assertEqual(len(self.plotter.axes), 1)
        self.assertListEqual(
            self.plotter.axes, [self.plotter.subplots['map'].ax]
        )
        self.plotter.del_subplot('map')
        self.assertEqual(len(self.plotter.axes), 0)
        self.assertListEqual(self.plotter.axes, [])

    def test_add_subplot_calls_subplot_plot_if_already_plotted(self):
        self.plotter.plot()
        self.assertEqual(len(self.plotter.axes), 1)
        self.assertListEqual(
            self.plotter.axes, [self.plotter.subplots['map'].ax]
        )
        self.plotter.add_subplot('test')
        self.assertTrue(self.plotter.subplots['test'].plotted)
        self.assertEqual(len(self.plotter.axes), 2)
        self.assertListEqual(
            self.plotter.axes,
            [self.plotter.subplots['map'].ax, self.plotter.subplots['test'].ax]
        )

    @patch('pylawr.plot.subplot.Subplot.plot')
    def test_plot_subplot_calls_plot_with_subplot_spec(self, p):
        self.plotter.new_figure()
        map_slices = self.plotter.grid_slices['map']
        subplot_spec = self.plotter.gridspec[map_slices[0], map_slices[1]]
        self.plotter._plot_subplot('map')
        p.assert_called_once()
        self.assertEqual(self.plotter.figure, p.call_args[1]['fig'])
        self.assertTupleEqual(subplot_spec.get_geometry(),
                              p.call_args[1]['spec'].get_geometry())

    def test_plot_subplot_raises_keyerror_if_not_available(self):
        with self.assertRaises(KeyError):
            self.plotter._plot_subplot('test')

    def test_plot_subplot_raises_notplotted_error(self):
        with self.assertRaises(NotPlottedError):
            self.plotter._plot_subplot('map')
        self.plotter.new_figure()
        self.plotter._plot_subplot('map')

    def test_change_grid_slice_plots_is_replotted_if_plotted(self):
        new_grid_slice = (slice(0, 1), slice(None, None))
        subplot_spec = self.plotter.gridspec[
            new_grid_slice[0], new_grid_slice[1]
        ]
        self.plotter.plot()
        with patch('pylawr.plot.subplot.Subplot.plot') as p:
            self.plotter.change_grid_slice('map', new_grid_slice)
            p.assert_called_once()
            self.assertEqual(self.plotter.figure, p.call_args[1]['fig'])
            self.assertTupleEqual(subplot_spec.get_geometry(),
                                  p.call_args[1]['spec'].get_geometry())

    def test_change_grid_slice_removes_plot_before_plotting_if_plotted(self):
        new_grid_slice = (slice(0, 1), slice(None, None))
        self.plotter.plot()
        self.assertEqual(len(self.plotter.figure.axes), 1)
        old_axes = self.plotter.subplots['map'].ax
        self.plotter.change_grid_slice('map', new_grid_slice)
        self.assertEqual(len(self.plotter.figure.axes), 1)
        self.assertNotEqual(id(self.plotter.subplots['map'].ax),
                            id(old_axes))

    @patch('pylawr.plot.subplot.Subplot.plot_layer_on_ax')
    def test_add_layer_calls_plot_layer_on_ax_if_plotted(self, p):
        layer = LawrHeaderLayer()
        self.plotter.plot()
        self.plotter.add_layer('map', layer)
        p.assert_called_once_with(layer=layer)

    def test_get_subplot_name_to_layer_checks_if_right_layer(self):
        with self.assertRaises(TypeError):
            self.plotter._get_subplot_name_from_layer(123.01)
            self.plotter._get_subplot_name_from_layer('test')
            self.plotter._get_subplot_name_from_layer([2435])

    def test_get_subplot_name_returns_subplot_name(self):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        self.plotter.add_subplot('test')
        self.plotter.subplots['map']._layers_list.append(layer)
        self.plotter.subplots['test']._layers_list.append(layer_1)
        returned_subplot_name = self.plotter._get_subplot_name_from_layer(layer)
        self.assertEqual(returned_subplot_name, 'map')
        returned_subplot_name = self.plotter._get_subplot_name_from_layer(
            layer_1)
        self.assertEqual(returned_subplot_name, 'test')

    def test_get_subplot_name_raises_keyerror_if_no_subplot_was_found(self):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        self.plotter.add_subplot('test')
        self.plotter.subplots['map']._layers_list.append(layer)
        with self.assertRaises(KeyError):
            self.plotter._get_subplot_name_from_layer(layer_1)

    def test_swap_layer_raises_argument_error_if_no_argument(self):
        layer = LawrHeaderLayer()
        with self.assertRaises(TypeError):
            self.plotter.swap_layer(layer)

    @patch('pylawr.plot.subplot.Subplot.swap_layer')
    def test_swap_layer_calls_swap_layer_from_determined_subplot(self, p):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        self.plotter.add_subplot('test')
        self.plotter.add_layer('map', layer)
        self.plotter.swap_layer(layer_1, layer)
        p.assert_called_once_with(new_layer=layer_1, old_layer=layer)

    @patch('pylawr.plot.subplot.Subplot.swap_layer')
    def test_swap_layer_calls_swap_layer_with_layer_to_layer_num(self, p):
        layer = LawrHeaderLayer()
        layer_1 = LawrHeaderLayer()
        self.plotter.add_subplot('test')
        self.plotter.add_layer('map', layer)
        self.plotter.swap_layer(layer_1, layer_num=('map', 0))
        p.assert_called_once_with(new_layer=layer_1, old_layer=layer)

    @patch('pylawr.plot.subplot.Subplot.swap_layer')
    def test_swap_layer_calls_swap_layer_with_layer_to_layer_zorder(self, p):
        layer = LawrHeaderLayer()
        layer.zorder = 1
        layer_1 = LawrHeaderLayer()
        self.plotter.add_subplot('test')
        self.plotter.add_layer('map', layer)
        self.plotter.swap_layer(layer_1, zorder=('map', 1))
        p.assert_called_once_with(new_layer=layer_1, old_layer=layer)


if __name__ == '__main__':
    unittest.main()
