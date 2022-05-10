#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np

# Internal modules
from .base_ops import Placeholder, Parameter


logger = logging.getLogger(__name__)


def gaussian_rbf(placeholder_name='distance', length_scale=200, stddev=1):
    """
    The gaussian, also called radial basis function, kernel is one of the
    default kernels for kernel based learning. It is universal and is can
    integrate into almost any function that you want. It has also infinitely
    many derivatives.

    Parameters
    ----------
    placeholder_name : str, optional
        The placeholder name, which is used to identify the placeholder in this
        kernel chain. Default name is `distance`.
    length_scale : float, optional
        This length scale is used within the gaussian kernel to define the
        decorrelation length of this kernel. Default is 200.
    stddev : float, optional
        The standard deviation of this gaussian kernel. The standard deviation
        defines the width gaussian distribution. Default is 1.

    Returns
    -------
    kernel : :py:class:`~pylawr.remap.kernel.base_ops.KernelNode`
        This is the precompiled gaussian kernel function, which can be evaluated
        with given `placeholder_name` and has given `length_scale` and `stddev`
        as trainable parameters.
    """
    placeholder_d = Placeholder(placeholder_name)
    param_l = Parameter(length_scale, name='decorrelation')
    param_stddev = Parameter(stddev, name='variance')

    inner_kernel = placeholder_d ** 2 / (2 * param_l ** 2)
    kernel = param_stddev ** 2 * np.exp(-inner_kernel)
    return kernel


def exp_sin_squared(placeholder_name='distance', length_scale=1.0,
                    periodicity=1.0):
    """
    The exponential sine squared kernel as in scikit-learn. This kernel can be
    used to model periodic processes.

    Parameters
    ----------
    placeholder_name : str, optional
        The placeholder name, which is used to identify the placeholder in this
        kernel chain. Default name is `distance`.
    length_scale : float, optional
        This length scale is used within this kernel to define the
        decorrelation length of this kernel. Default is 200.
    periodicity : float, optional
        The periodicity of this kernel. The periodicity defines the period of
        this kernel. Default is 1.

    Returns
    -------
    kernel : :py:class:`~pylawr.remap.kernel.base_ops.KernelNode`
        This is the precompiled exponential sine squared kernel, which can be
        evaluated with given `placeholder_name` and has given `length_scale` and
        `periodicity` as trainable parameters.
    """
    placeholder_d = Placeholder(placeholder_name)
    param_l = Parameter(length_scale, name='decorrelation')
    param_period = Parameter(periodicity, name='periodicity')

    sin_period = np.sin(np.pi / param_period * placeholder_d)
    inner_kernel = -2 * (sin_period / param_l) ** 2
    kernel = np.exp(inner_kernel)
    return kernel


def rational_quadratic(placeholder_name='distance', length_scale=200, stddev=1,
                       scale=1):
    """
    This rational quadratic kernel is a mixture of gaussian / radial basis
    function kernels. This mixture combines these kernel with different
    length scales with the scale parameter as relative weighting between
    large-scale and small-scale variations.

    Parameters
    ----------
    placeholder_name : str, optional
        The placeholder name, which is used to identify the placeholder in this
        kernel chain. Default name is `distance`.
    length_scale : float, optional
        This length scale is used within the rational kernel to define the
        decorrelation length of this kernel. Default is 200.
    stddev : float, optional
        The standard deviation of this rational kernel. The standard deviation
        defines the width gaussian distribution. Default is 1.
    scale : float, optional
        This scale parameter is a weighting between large- and small-scale
        variations. If this factor converges towards zero, only large scale
        variations are important for the weights of kriging and every distance
        will be then unity. If this factor converges toward infinity, this
        kernel is the same as the gaussian rbf kernel. Default is 1.

    Returns
    -------
    kernel : :py:class:`~pylawr.remap.kernel.base_ops.KernelNode`
        This is the precompiled rational quadratic kernel, which can be
        evaluated with given `placeholder_name` and has given `length_scale`,
        `stddev` and `scale` as trainable parameters.
    """
    placeholder_d = Placeholder(placeholder_name)
    param_l = Parameter(length_scale, name='decorrelation')
    param_stddev = Parameter(stddev, name='variance')
    param_scale = Parameter(scale, name='scaling')

    inner_kernel = 1 + placeholder_d ** 2 / (2 * param_l * param_scale)
    kernel = (param_stddev ** 2) * (inner_kernel ** (-param_scale))
    return kernel
