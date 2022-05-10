#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import os
from unittest.mock import MagicMock, PropertyMock
import functools

# External modules
import numpy as np
import xarray as xr

# Internal modules
from pylawr.remap.kernel.base_ops import *


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestBaseKernel(unittest.TestCase):
    def test_init_sets_default_values_for_placeholder_params(self):
        kernel = BaseKernel()
        self.assertIsInstance(kernel.params, list)
        self.assertFalse(kernel.params)
        self.assertIsInstance(kernel.placeholders, list)
        self.assertFalse(kernel.placeholders)

    def test_base_kernel_inherits_from_np_mixin(self):
        kernel = BaseKernel()
        self.assertIsInstance(kernel, np.lib.mixins.NDArrayOperatorsMixin)

    def test_array_ufunc_pops_out(self):
        kernel = Parameter(10, constant=True)
        ret_kernel = np.sin(kernel, out=10)
        self.assertNotIn('out', ret_kernel.np_func.keywords.keys())

    def test_array_raises_not_implemented_error_if_not_called(self):
        kernel = Parameter(10, constant=True)
        with self.assertRaises(NotImplementedError):
            kernel.__array_ufunc__(np.sin, '__test__',)

    def test_array_ufunc_partial_fills_func(self):
        kernel = Parameter(10, constant=True)
        ret_kernel = np.sin(kernel, where=(False, ))
        self.assertIsInstance(ret_kernel.np_func, functools.partial)
        self.assertEqual(ret_kernel.np_func.func, np.sin)
        self.assertIn('where', ret_kernel.np_func.keywords.keys())

    def test_array_ufunc_converts_data_into_parameter(self):
        kernel = Parameter(10, constant=True)
        test_array = np.array((10, 10))
        ret_kernel = kernel * test_array
        self.assertIsInstance(ret_kernel.dependencies[1], Parameter)
        np.testing.assert_equal(
            ret_kernel.dependencies[1].value,
            test_array
        )
        self.assertTrue(ret_kernel.dependencies[1].constant)

    def test_array_ufunc_returns_kernel_node_with_inputs(self):
        kernel = Parameter(10, constant=True)
        test_array = Parameter(np.array((10, 10)), constant=True)
        ret_kernel = kernel * test_array
        self.assertIsInstance(ret_kernel, KernelNode)
        self.assertTupleEqual(ret_kernel.dependencies, (kernel, test_array))

    def test_transpose_returns_kernel_node_with_transpose(self):
        kernel = Parameter(np.arange(9).reshape(3, 3), constant=True)
        ret_kernel = kernel.transpose()
        self.assertIsInstance(ret_kernel, KernelNode)
        self.assertEqual(ret_kernel.np_func, np.transpose)

    def test_T_calls_transpose(self):
        kernel = Parameter(np.arange(9).reshape(3, 3), constant=True)
        kernel.transpose = MagicMock()
        _ = kernel.T
        kernel.transpose.assert_called_once()


class TestPlaceholder(unittest.TestCase):
    def test_placeholder_sets_name(self):
        placeholder = Placeholder('test')
        self.assertEqual(placeholder.name, 'test')

    def test_placeholder_returns_list_of_self_as_placeholder(self):
        placeholder = Placeholder('test')
        self.assertListEqual(placeholder.placeholders, [placeholder, ])

    def test_placeholders_returns_args_if_given(self):
        placeholder = Placeholder('test')
        return_val = placeholder(12, test=24)
        self.assertEqual(return_val, 12)

    def test_placeholders_returns_kwargs_if_given(self):
        placeholder = Placeholder('test')
        return_val = placeholder(test=24)
        self.assertEqual(return_val, 24)

    def test_placeholders_raises_valueserror(self):
        placeholder = Placeholder('test')
        with self.assertRaises(ValueError):
            _ = placeholder()
        with self.assertRaises(ValueError):
            _ = placeholder(bla=12)

    def test_params_are_empty(self):
        placeholder = Placeholder('test')
        self.assertFalse(placeholder.params)

    def test_repr_returns_placeholder_str(self):
        placeholder = Placeholder('test')
        self.assertEqual(
            repr(placeholder),
            'placeholder("test")'
        )


class TestParameter(unittest.TestCase):
    def test_init_start_value_sets_value(self):
        parameter = Parameter(value=10)
        self.assertEqual(parameter.value, 10)
        self.assertFalse(parameter.constant)

    def test_init_sets_constant(self):
        parameter = Parameter(value=10, constant=True)
        self.assertTrue(parameter.constant)
        parameter = Parameter(value=10, constant=False)
        self.assertFalse(parameter.constant)

    def test_call_returns_value(self):
        parameter = Parameter(value=10, constant=True)
        ret_val = parameter(20, test=30)
        self.assertEqual(ret_val, 10)

    def test_params_returns_list_of_self_if_not_constant(self):
        parameter = Parameter(value=10, constant=False)
        self.assertListEqual(parameter.params, [parameter, ])
        parameter.constant = True
        self.assertFalse(parameter.params)

    def test_placeholders_returns_empty_list(self):
        parameter = Parameter(value=10, constant=False)
        self.assertFalse(parameter.placeholders)

    def test_repr_not_constant(self):
        parameter = Parameter(value=10, constant=False)
        self.assertEqual(
            repr(parameter),
            'parameter(10)'
        )

    def test_repr_constant(self):
        parameter = Parameter(value=10, constant=True)
        self.assertEqual(
            repr(parameter),
            'constant(10)'
        )

    def test_name_gets_private(self):
        parameter = Parameter(value=10, constant=False)
        parameter._name = 'test'
        self.assertEqual(parameter.name, 'test')

    def test_name_sets_private(self):
        parameter = Parameter(value=10, constant=False)
        self.assertIsNone(parameter._name)
        parameter.name = 'test'
        self.assertEqual(parameter._name, 'test')
        parameter.name = None
        self.assertIsNone(parameter._name)

    def test_name_raises_type_error_if_not_str_or_none(self):
        parameter = Parameter(value=10, constant=False)
        with self.assertRaises(TypeError):
            parameter.name = 12


class TestSingleDepKernel(unittest.TestCase):
    def test_init_sets_other_to_other(self):
        other = BaseKernel()
        kernel = KernelNode(other)
        self.assertEqual(id(kernel.dependencies[0]), id(other))

    def test_init_sets_placeholders_to_other_placeholders(self):
        other = MagicMock()
        type(other).placeholders = PropertyMock(return_value=[12, ])
        kernel = KernelNode(other)
        self.assertListEqual(kernel.placeholders, other.placeholders)

    def test_init_sets_params_to_other_params(self):
        other = MagicMock()
        type(other).params = PropertyMock(return_value=[12, ])
        kernel = KernelNode(other)
        self.assertEqual(kernel.params, other.params)

    def test_call_uses_ufunc(self):
        other = Parameter(np.random.normal(size=(100, 100)))
        right_value = np.exp(other.value)
        kernel = KernelNode(other)
        kernel._np_func = np.exp
        returned_value = kernel()
        np.testing.assert_equal(returned_value, right_value)

    def test_repr_dep_node(self):
        other = Parameter(10)
        ret_kernel = np.exp(other)
        repr_str = 'exp({0:s})'.format(repr(other))
        self.assertEqual(
            repr(ret_kernel),
            repr_str
        )

    def test_np_func_raises_attribute_error_if_not_set(self):
        kernel = KernelNode()
        with self.assertRaises(AttributeError):
            _ = kernel.np_func()


class TestDoubleDepKernel(unittest.TestCase):
    def test_init_sets_others(self):
        other_1 = MagicMock()
        type(other_1).placeholders = PropertyMock(return_value=[12, ])
        other_2 = MagicMock()
        type(other_2).placeholders = PropertyMock(return_value=[24, ])
        kernel = KernelNode(other_1, other_2)
        self.assertEqual(id(kernel.dependencies[0]), id(other_1))
        self.assertEqual(id(kernel.dependencies[1]), id(other_2))

    def test_init_sets_placeholders_to_other_placeholders(self):
        other_1 = MagicMock()
        type(other_1).placeholders = PropertyMock(return_value=[12, ])
        other_2 = MagicMock()
        type(other_2).placeholders = PropertyMock(return_value=[24, ])
        kernel = KernelNode(other_1, other_2)
        self.assertListEqual(kernel.placeholders, [12, 24])

    def test_init_sets_params_to_other_params(self):
        other_1 = MagicMock()
        type(other_1).params = PropertyMock(return_value=[12, ])
        other_2 = MagicMock()
        type(other_2).params = PropertyMock(return_value=[24, ])
        kernel = KernelNode(other_1, other_2)
        self.assertListEqual(kernel.params, [12, 24])

    def test_call_uses_ufunc(self):
        other_1 = Parameter(np.random.normal(size=(100, 100)))
        other_2 = Parameter(10)
        right_value = np.add(other_1.value, other_2.value)
        kernel = KernelNode(other_1, other_2)
        kernel._np_func = np.add
        returned_value = kernel()
        np.testing.assert_equal(returned_value, right_value)

    def test_get_named_parameter_returns_found_params(self):
        other_1 = Parameter(10, name='decorrelation')
        other_2 = Parameter(10, name='test')
        kernel = other_1 + other_2
        self.assertListEqual(kernel.get_named_param('decorrelation'),
                             [other_1, ])

    def test_get_named_parameter_returns_all_found_params(self):
        other_1 = Parameter(10, name='decorrelation')
        kernel = other_1 + other_1
        self.assertListEqual(kernel.get_named_param('decorrelation'),
                             [other_1, other_1])

    def test_get_named_parameter_raises_type_error_if_name_not_str(self):
        other_1 = Parameter(10, name='decorrelation')
        with self.assertRaises(TypeError):
            other_1.get_named_param(None)
        with self.assertRaises(TypeError):
            other_1.get_named_param(12)


if __name__ == '__main__':
    unittest.main()
