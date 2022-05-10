#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
import importlib

# External modules

# Internal modules
from pylawr.plot.backend import BackendLoader


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestBackendLoader(unittest.TestCase):
    def setUp(self):
        self.backend = BackendLoader('agg')

    def test_name_gets_private_name(self):
        self.assertEqual(self.backend.name, self.backend._name)
        self.backend._name = 'tk'
        self.assertEqual(self.backend.name, self.backend._name)

    def test_name_sets_private_name(self):
        self.backend._name = None
        self.backend.name = 'tkagg'
        self.assertEqual(self.backend.name, self.backend._name)
        self.assertEqual(self.backend._name, 'TkAgg')

    def test_name_setter_checks_available_names(self):
        self.backend._name = None
        with self.assertRaises(ValueError) as e:
            self.backend.name = 'qt'
        self.assertIsNone(self.backend._name)
        self.backend.name = 'tkagg'
        self.assertNotEqual(self.backend._name, 'tkagg')
        self.assertEqual(self.backend._name, 'TkAgg')

    def test_load_backend_sets_backend_to_mpl_module(self):
        module_name = '{0:s}_pdf'.format(self.backend._backend_path_template)
        right_module = importlib.import_module(module_name)
        self.backend.name = 'pdf'
        self.backend._backend = None
        self.backend._load_backend()
        self.assertIsNotNone(self.backend._backend)
        self.assertEqual(self.backend._backend, right_module)

    def test_name_setter_loads_module(self):
        module_name = '{0:s}_pdf'.format(self.backend._backend_path_template)
        right_module = importlib.import_module(module_name)
        self.backend._backend = None
        self.assertNotEqual(self.backend._backend, right_module)
        self.backend.name = 'pdf'
        self.assertIsNotNone(self.backend._backend)
        self.assertEqual(self.backend._backend, right_module)

    def test_canvas_returns_backend_canvas(self):
        self.assertIsNotNone(self.backend.canvas)
        module_name = '{0:s}_pdf'.format(self.backend._backend_path_template)
        right_module = importlib.import_module(module_name)
        right_canvas = getattr(right_module, 'FigureCanvas')
        self.assertNotEqual(self.backend.canvas, right_canvas)
        self.backend._backend = right_module
        self.assertEqual(self.backend.canvas, right_canvas)


if __name__ == '__main__':
    unittest.main()
