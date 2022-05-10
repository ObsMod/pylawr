#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules

# Internal modules


logger = logging.getLogger(__name__)


class KernelVariogram(object):
    """
    The kernel variogram is an observation operator, which transforms a set of
    parameters into a pre-defined kernel. The variogram of this kernel is then
    evaluated based on given observation distances. The order of the parameters
    should be the same between kernel and given parameters.

    Parameters
    ----------
    kernel : child of :py:class:`~pylawr.remap.kernel.base_ops.BaseKernel`
        This kernel is used as basis kernel for the observation operator. The
        given parameters are set as parameters of this kernel.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, params, obs_dist, **kwargs):
        """
        This call evaluates the kernel with given parameters as observation
        distances.

        Parameters
        ----------
        params : iterable
            These parameters are set as parameters of set kernel. It is assumed
            that the order of parameter is the same for kernel and this
            argument.
        obs_dist : :py:class:`numpy.ndarray`
            The theoretical variogram of the kernel is estimated for these given
            observation distances.
        kwargs : dict
            These additional keyword arguments are not used in this observation
            operator.

        Returns
        -------
        variogram : :py:class:`numpy.ndarray`
            The theoretical variogram of set kernel, conditioned on given
            parameters and observation distances.
        """
        for i, p_inst in enumerate(self.kernel.params):
            p_inst.value = params[i]
        variogram = self.kernel.variogram(obs_dist)
        return variogram
