#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch
from collections import OrderedDict

# External modules
import numpy as np

import matplotlib.figure as mpl_figure
import matplotlib.lines as mpl_lines
import matplotlib.text as mpl_text

# Internal modules
from pylawr.plot.layer.lawr_header import LawrHeaderLayer


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestHeaderLayer(unittest.TestCase):
    def setUp(self):
        self.layer = LawrHeaderLayer()
        self.figure = mpl_figure.Figure()
        self.axes = self.figure.add_subplot(111)

    @patch('matplotlib.axes.Axes.axhline')
    def test_plot_lines_creates_a_horizontal_line(self, p):
        _ = self.layer._plot_lines_on_ax(self.axes)
        p.assert_called_once_with(y=2/3, xmin=0, xmax=1, color='k', zorder=0,
                                  lw=0.5)

    @patch('matplotlib.axes.Axes.axvline')
    def test_plot_lines_creates_a_vertical_line(self, p):
        _ = self.layer._plot_lines_on_ax(self.axes)
        p.assert_called_once_with(x=1/2, ymin=0, ymax=2/3, color='k', lw=0.5,
                                  zorder=0)

    def test_plot_lines_returns_both_lines(self):
        right_h = self.axes.axhline(y=2/3, xmin=0, xmax=1, color='k')
        right_v = self.axes.axvline(x=1/2, ymin=0, ymax=2/3, color='k')
        line_h, line_v = self.layer._plot_lines_on_ax(self.axes)
        self.assertIsInstance(line_h, mpl_lines.Line2D)
        self.assertIsInstance(line_v, mpl_lines.Line2D)
        np.testing.assert_equal(right_h.get_xydata(), line_h.get_xydata())
        np.testing.assert_equal(right_v.get_xydata(), line_v.get_xydata())

    def test_plot_lines_sets_zorder(self):
        self.layer.zorder = 0
        line_h, line_v = self.layer._plot_lines_on_ax(self.axes)
        self.assertEqual(line_h.zorder, 0)
        self.assertEqual(line_v.zorder, 0)
        self.layer.zorder = 10
        line_h, line_v = self.layer._plot_lines_on_ax(self.axes)
        self.assertEqual(line_h.zorder, 10)
        self.assertEqual(line_v.zorder, 10)

    def test_plot_lines_uses_line_settings(self):
        self.layer.line_settings = dict(lw=10)
        line_h, line_v = self.layer._plot_lines_on_ax(self.axes)
        self.assertEqual(line_h.get_lw(), 10)
        self.assertEqual(line_v.get_lw(), 10)
        self.layer.line_settings = dict(ls='--')
        line_h, line_v = self.layer._plot_lines_on_ax(self.axes)
        self.assertNotEqual(line_h.get_lw(), 10)
        self.assertNotEqual(line_v.get_lw(), 10)
        self.assertEqual(line_h.get_ls(), '--')
        self.assertEqual(line_v.get_ls(), '--')

    @patch('matplotlib.axes.Axes.text')
    def test_plot_text_on_ax_creates_a_text(self, p):
        ax_coords = self.axes.transAxes
        _ = self.layer._plot_text_on_ax(self.axes, x=0, y=0, text='bla')
        p.assert_called_once_with(
            x=0, y=0, s='bla', zorder=0, transform=ax_coords
        )

    def test_plot_text_on_ax_returns_plotted_text_object(self):
        ret_text = self.layer._plot_text_on_ax(self.axes, x=0, y=0, text='bla')
        self.assertIsInstance(ret_text, mpl_text.Text)
        right_text = self.axes.get_children()[0]
        self.assertEqual(right_text, ret_text)

    def test_plot_text_uses_given_text(self):
        ret_text = self.layer._plot_text_on_ax(self.axes, x=0, y=0, text='bla')
        self.assertEqual(ret_text.get_text(), 'bla')
        ret_text = self.layer._plot_text_on_ax(self.axes, x=0, y=0, text='test')
        self.assertEqual(ret_text.get_text(), 'test')

    def test_plot_text_passes_zorder_to_text(self):
        self.layer.zorder = 2
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        self.assertEqual(ret_text.zorder, self.layer.zorder)
        self.layer.zorder = 42
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        self.assertEqual(ret_text.zorder, 42)

    def test_plot_text_uses_trans_axes_for_transform(self):
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        ax_coords = self.axes.transAxes
        self.assertEqual(ret_text.get_transform(), ax_coords)

    def test_plot_text_uses_x_y_from_args(self):
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        self.assertTupleEqual(ret_text.get_position(), (0, 0))
        ret_text = self.layer._plot_text_on_ax(self.axes, 10, 0, 'bla')
        self.assertTupleEqual(ret_text.get_position(), (10, 0))
        ret_text = self.layer._plot_text_on_ax(self.axes, 10, 100, 'bla')
        self.assertTupleEqual(ret_text.get_position(), (10, 100))

    def test_plot_text_uses_settings_from_layer(self):
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        old_color = ret_text.get_color()
        self.layer['color'] = '0.5'
        self.assertNotEqual(old_color, '0.5')
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        self.assertNotEqual(old_color, ret_text.get_color())
        self.assertEqual(ret_text.get_color(), '0.5')

        self.layer.settings = {'bla': 123, 'test': 234, 'hp': 764}
        ax_coords = self.axes.transAxes
        with patch('matplotlib.axes.Axes.text') as p:
            _ = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
            p.assert_called_once_with(
                x=0, y=0, s='bla', zorder=0, transform=ax_coords,
                **self.layer.settings
            )

    def test_plot_text_updates_passed_settings_with_specific_settings(self):
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        old_color = ret_text.get_color()
        self.assertNotEqual(old_color, '0.5')
        ret_text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla',
                                               color='0.5')
        self.assertNotEqual(old_color, ret_text.get_color())
        self.assertEqual(ret_text.get_color(), '0.5')

        self.layer.settings['color'] = '0.1'
        settings = {'bla': 123, 'test': 234, 'hp': 764}
        ax_coords = self.axes.transAxes
        with patch('matplotlib.axes.Axes.text') as p:
            _ = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla', **settings)
            p.assert_called_once_with(
                x=0, y=0, s='bla', zorder=0, transform=ax_coords,
                color='0.1', **settings
            )

    def test_plot_title_uses_title_as_text(self):
        text_title = self.layer._plot_title_on_ax(self.axes)
        self.assertEqual(text_title.get_text(), self.layer.title)
        self.layer.title = 'test title'
        text_title = self.layer._plot_title_on_ax(self.axes)
        self.assertEqual(text_title.get_text(), 'test title')

    def test_plot_title_uses_title_settings(self):
        self.layer.title_settings = dict(
            color='0.5'
        )
        text_title = self.layer._plot_title_on_ax(self.axes)
        self.assertEqual(text_title.get_color(), '0.5')
        self.layer.title_settings = dict(
            va='top'
        )
        text_title = self.layer._plot_title_on_ax(self.axes)
        self.assertNotEqual(text_title.get_color(), '0.5')
        self.assertEqual(text_title.get_verticalalignment(), 'top')

    @patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer._plot_text_on_ax',
           return_value=123)
    def test_plot_title_calls_text_on_ax(self, p):
        ret_text = self.layer._plot_title_on_ax(self.axes)
        self.assertEqual(ret_text, 123)
        p.assert_called_once_with(
            ax=self.axes, x=0.5, y=5/6, text=self.layer.title,
            **self.layer.title_settings
        )

    @patch('numpy.linspace', return_value=[0, 1/6, 2/6, 3/6, 4/6])
    def test_info_box_on_ax_creates_lin_space_for_y_pos(self, p):
        info_dict = dict(
            a='1',
            b='2',
            c='3'
        )
        _ = self.layer._plot_info_box_on_ax(ax=self.axes, info_dict=info_dict,
                                            x_pos=0)
        p.assert_called_once_with(2/3, 0, 5)

    def test_info_box_returns_list_with_texts(self):
        info_dict = dict(
            a='1',
            b='2',
            c='3'
        )
        test_list = self.layer._plot_info_box_on_ax(
            ax=self.axes, info_dict=info_dict, x_pos=0
        )
        self.assertIsInstance(test_list, list)
        for ele in test_list:
            self.assertIsInstance(ele, mpl_text.Text)

    def test_info_box_calls_plot_text_for_every_element(self):
        text = self.layer._plot_text_on_ax(self.axes, 0, 0, 'bla')
        info_dict = OrderedDict(
            a='1',
            b='2',
            c='3'
        )
        trg = 'pylawr.plot.layer.lawr_header.LawrHeaderLayer._plot_text_on_ax'
        with patch(trg, return_value=text) as p:
            _ = self.layer._plot_info_box_on_ax(
                ax=self.axes, info_dict=info_dict, x_pos=0
            )
            p.assert_called()
            self.assertEqual(p.call_count, 3)

    def test_info_box_uses_info_settings(self):
        info_dict = OrderedDict(
            a='1',
        )
        text = '{0:<5}: {1:<40}'.format('a', info_dict['a'])
        y_pos = np.linspace(2 / 3, 0, len(info_dict) + 2)[1:-1]
        self.layer.info_settings = {'color': '0.5', 'alpha': 0.5}
        trg = 'pylawr.plot.layer.lawr_header.LawrHeaderLayer._plot_text_on_ax'
        with patch(trg, return_value=1) as p:
            _ = self.layer._plot_info_box_on_ax(
                ax=self.axes, info_dict=info_dict, x_pos=0
            )
            p.assert_called_once_with(
                ax=self.axes, x=0, y=y_pos[0], text=text,
                **self.layer.info_settings
            )

    @patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer._plot_lines_on_ax',
           return_value=(1, 2))
    def test_plot_calls_plot_lines(self, p):
        self.layer.plot(ax=self.axes)
        p.assert_called_once_with(ax=self.axes)

    @patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer._plot_title_on_ax',
           return_value=1)
    def test_plot_calls_plot_title(self, p):
        self.layer.plot(ax=self.axes)
        p.assert_called_once_with(ax=self.axes)

    @patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer._plot_info_box_on_ax',
           return_value=[1, ])
    def test_plot_calls_plot_info_for_left(self, p):
        self.layer.text_padding = 0.1
        self.layer.plot(ax=self.axes)
        call_args = dict(ax=self.axes, info_dict=self.layer.left,
                         x_pos=0+self.layer.text_padding)
        self.assertDictEqual(p.call_args_list[0][1], call_args)

    @patch('pylawr.plot.layer.lawr_header.LawrHeaderLayer._plot_info_box_on_ax',
           return_value=[1, ])
    def test_plot_calls_plot_info_for_right(self, p):
        self.layer.right = OrderedDict(bla='test')
        self.layer.text_padding = 0.1
        self.layer.plot(ax=self.axes)
        call_args = dict(ax=self.axes, info_dict=self.layer.right,
                         x_pos=0.5+self.layer.text_padding)
        self.assertDictEqual(p.call_args_list[1][1], call_args)

    def test_plot_adds_collected_plotted_elements_to_collection(self):
        self.layer.left = {}
        self.layer.right = {}
        self.assertFalse(self.layer.collection)
        self.layer.plot(self.axes)
        self.assertTrue(self.layer.collection)
        self.assertEqual(len(self.layer.collection), 3)
        self.layer.left = {'test': 'test'}
        self.layer.plot(self.axes)
        self.assertEqual(len(self.layer.collection), 7)


if __name__ == '__main__':
    unittest.main()
