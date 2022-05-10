#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os

# External modules

# Internal modules
from pylawr.utilities.decorators import tuplesetter


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class GetterSetterClass(object):
    def __init__(self):
        self._test = None

    @property
    def test(self):
        return self._test

    @tuplesetter(test, len_tuple=2)
    def test(self, new_test):
        return new_test


class TestTupleSetter(unittest.TestCase):
    def setUp(self):
        self.cls = GetterSetterClass()
        self._tval = None

    @property
    def tval(self):
        return self._tval

    def test_decorator_raises_valueerror_if_fn_has_only_one_arg(self):
        err_msg = "`tuplesetter` decorator can only be used for methods that " \
                  "take the object reference as first argument and the new " \
                  "property value as second argument!"

        with self.assertRaises(ValueError) as e:
            class NewGetterSetterClass(object):
                def __init__(self):
                    self._test = None

                @property
                def test(self):
                    return self._test

                @tuplesetter(test, len_tuple=2)
                def test(self):
                    return self

        self.assertEqual(err_msg, str(e.exception))

    def test_decorator_raises_valueerror_if_fn_has_more_than_two_args(self):
        err_msg = "`tuplesetter` decorator can only be used for methods that " \
                  "take the object reference as first argument and the new " \
                  "property value as second argument!"

        with self.assertRaises(ValueError) as e:
            class NewGetterSetterClass(object):
                def __init__(self):
                    self._test = None

                @property
                def test(self):
                    return self._test

                @tuplesetter(test, len_tuple=2)
                def test(self, text, a, b, c, d):
                    return self

        self.assertEqual(err_msg, str(e.exception))

    def test_decorator_raises_valueerror_if_fn_dosnt_ref_first_arg(self):
        err_msg = "`tuplesetter` decorator can only be used for methods that " \
                  "take the object reference as first argument, which needs " \
                  "to be called `self`!"

        with self.assertRaises(ValueError) as e:
            class NewGetterSetterClass(object):
                def __init__(self):
                    self._test = None

                @property
                def test(self):
                    return self._test

                @tuplesetter(test, len_tuple=2)
                def test(msg, text):
                    return text

        self.assertEqual(err_msg, str(e.exception))

    def test_tuple_setter_sets_private(self):
        self.assertIsNone(self.cls._test)
        self.cls.test = (2, 3)
        self.assertTupleEqual((2, 3), self.cls._test)

    def test_tuple_setter_converts_iterable_into_tuple(self):
        self.assertIsNone(self.cls._test)
        self.cls.test = [2, 3]
        self.assertIsInstance(self.cls._test, tuple)
        self.assertTupleEqual((2, 3), self.cls._test)

    def test_other_converted_into_tuple(self):
        self.assertIsNone(self.cls._test)
        self.cls.test = 3
        self.assertIsInstance(self.cls._test, tuple)
        self.assertTupleEqual((3, 3), self.cls._test)

    def test_other_number_of_elements_equals_len_tuple(self):
        len_tuple = 5

        class NewGetterSetterClass(object):
            def __init__(self):
                self._test = None

            @property
            def test(self):
                return self._test

            @tuplesetter(test, len_tuple=len_tuple)
            def test(self, newval):
                return newval
        test_cls = NewGetterSetterClass()
        test_cls.test = 3
        self.assertEqual(len(test_cls.test), len_tuple)
        self.assertTrue(all([a == 3for a in test_cls.test]))

    def test_iterable_is_shortened(self):
        test_list = [1, 3, 4, 5, 6]
        self.cls.test = test_list
        self.assertTupleEqual(tuple(test_list[:2]), self.cls.test)

    def test_iterable_too_short_raise_value_error(self):
        err_msg = "The given new value is too short! " \
                  "desired length: 2, actual length: 1"
        with self.assertRaises(ValueError) as e:
            self.cls.test = [1, ]
        self.assertEqual(err_msg, str(e.exception))

    def test_none_is_set_as_none(self):
        self.cls.test = (2, 3, 4)
        self.assertIsNotNone(self.cls._test)
        self.cls.test = None
        self.assertIsNone(self.cls._test)

    def test_str_is_set_like_int(self):
        self.cls.test = 'a'
        self.assertTupleEqual(('a', 'a'), self.cls._test)

    def test_raises_type_error_if_tuple_type_not_valid(self):
        err_msg = "Not all elements of the given tuple have the right type!"

        class NewGetterSetterClass(object):
            def __init__(self):
                self._test = None

            @property
            def test(self):
                return self._test

            @tuplesetter(test, valid_types=int)
            def test(self, newval):
                return newval
        test_cls = NewGetterSetterClass()
        with self.assertRaises(TypeError) as e:
            test_cls.test = 3.3
        self.assertEqual(err_msg, str(e.exception))


if __name__ == '__main__':
    unittest.main()
