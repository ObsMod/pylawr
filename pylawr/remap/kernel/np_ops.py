#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np

# Internal modules
from .base_ops import BaseKernel


logger = logging.getLogger(__name__)


class WhiteNoise(BaseKernel):
    def __init__(self, dependency, noise_level=1.0, constant=False):
        """
        This white noise kernel specifies the observation uncertainty. This
        white noise kernel returns a diagonal matrix with given noise level on
        the diagonal. Given `dependency` matrix is used to determine the shape
        of the resulting matrix. White noise is uncorrelated and
        stationary random noise (no bias) with the specified noise level as
        variance. If added to another kernel, this kernel acts as Tikhonov
        regularization of the kriging solution.

        Parameters
        ----------
        dependency :
        child of :py:class:`~pylawr.remap.kernel.base_ops.BaseKernel`
            The shape of the resulting covariance is estimated based on the
            output of  this parent kernel node. Normally, one would specify a
            :py:class:`~pylawr.remap.kernel.base_ops.Placeholder` as parent
            kernel node.
        noise_level : float, optional
            This noise level is set a diagonal elements of the covariance.
            Default is 1.
        constant : bool, optional
            If this white noise can be tuned (false) during optimization or if
            it is constant (True). Default False.
        """
        self.dependency = dependency
        self._noise_level = None
        self.noise_level = noise_level
        self.constant = constant
        self.name = 'WhiteNoise'

    @property
    def value(self):
        return self._noise_level

    @value.setter
    def value(self, new_value):
        self.noise_level = new_value

    @property
    def noise_level(self):
        return self._noise_level

    @noise_level.setter
    def noise_level(self, new_level):
        self._noise_level = new_level

    def __call__(self, *args, **kwargs):
        dep_val = self.dependency(*args, **kwargs)
        dep_shape = dep_val.shape
        if len(dep_shape) < 2 or dep_shape[-2] != dep_shape[-1]:
            noise_array = np.zeros_like(dep_val)
        else:
            noise_array = self.noise_level * np.eye(dep_shape[-1])
            noise_array = np.broadcast_to(noise_array, dep_shape).copy()
        return noise_array

    def diag(self, *args, **kwargs):
        dep_val = self.dependency.diag(*args, **kwargs)
        dep_shape = dep_val.shape
        diag_array = np.full(dep_shape, self.noise_level)
        return diag_array

    @property
    def params(self):
        if self.constant:
            return self.dependency.params
        else:
            return self.dependency.params + [self, ]

    @property
    def placeholders(self):
        return self.dependency.placeholders
