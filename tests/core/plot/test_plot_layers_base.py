#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch

# External modules
import matplotlib.figure as mpl_figure

# Internal modules
from pylawr.plot.layer.base import BaseLayer


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestBaseLayer(unittest.TestCase):
    def setUp(self):
        self.layer = BaseLayer()

    def test_layer_acts_like_dict_for_settings(self):
        self.layer.settings = {'test': 123}
        self.assertEqual(self.layer['test'], 123)
        self.layer['test1'] = 123
        self.assertDictEqual(self.layer.settings,
                             {'test': 123, 'test1': 123})
        self.layer['test'] = 1
        self.assertDictEqual(self.layer.settings,
                             {'test': 1, 'test1': 123})
        del self.layer['test1']
        self.assertDictEqual(self.layer.settings,
                             {'test': 1})
        self.layer.update({'test1': 123})
        self.assertDictEqual(self.layer.settings,
                             {'test': 1, 'test1': 123})

    def test_collection_returns_private_collection(self):
        self.assertListEqual(self.layer.collection, self.layer._collection)
        self.layer._collection = [123, 345]
        self.assertListEqual(self.layer.collection, [123, 345])
        self.layer._collection = []
        self.assertFalse(self.layer.collection)

    def test_remove_iterates_through_collection_and_calls_remove(self):
        figure = mpl_figure.Figure()
        ax = figure.add_subplot(111)
        text = ax.text(1, 1, 'bla')
        arrow = ax.arrow(1, 1, 0.3, 0.3)
        self.layer._collection = [text, arrow]
        with patch('matplotlib.text.Text.remove') as t_patch:
            with patch('matplotlib.patches.FancyArrow.remove') as a_patch:
                self.layer.remove()
                a_patch.assert_called_once()
                t_patch.assert_called_once()

    def test_remove_empties_collection(self):
        figure = mpl_figure.Figure()
        ax = figure.add_subplot(111)
        text = ax.text(1, 1, 'bla')
        arrow = ax.arrow(1, 1, 0.3, 0.3)
        self.layer._collection = [text, arrow]
        self.assertTrue(self.layer._collection)
        self.layer.remove()
        self.assertFalse(self.layer._collection)


if __name__ == '__main__':
    unittest.main()
