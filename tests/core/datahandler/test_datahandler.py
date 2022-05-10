#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import os
import unittest

# Internal modules

# External modules
from pylawr.datahandler.base import DataHandler
from pylawr.grid import GridNotAvailableError


logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL',logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestFileHandler(unittest.TestCase):
    def setUp(self):
        self.file_handler = open(os.path.join(DATA_PATH, 'lawr_data.txt'),
                                 mode='rb')
        self.data_handler = DataHandler(self.file_handler)

    def tearDown(self):
        self.file_handler.close()

    def test_fh_returns_private_method(self):
        self.assertEqual(id(self.data_handler.fh), id(self.data_handler._fh))

    def test_fh_sets_private(self):
        self.data_handler._fh = None
        self.data_handler.fh = self.file_handler
        self.assertEqual(id(self.file_handler), id(self.data_handler._fh))

    def test_fh_raises_error_if_not_readable(self):
        with self.assertRaises(TypeError):
            self.data_handler.fh = None
        self.file_handler.close()
        with self.assertRaises(ValueError):
            self.data_handler.fh = self.file_handler

    def test_close_sets_data_to_none(self):
        self.data_handler._data = 'test'
        self.assertEqual(self.data_handler._data, 'test')
        self.data_handler.close()
        self.assertIsNone(self.data_handler._data)

    def test_get_grid_raises_GridNotAvailableError(self):
        with self.assertRaises(GridNotAvailableError):
            _ = self.data_handler.grid


if __name__ == '__main__':
    unittest.main()
