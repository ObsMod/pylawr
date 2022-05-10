#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import scipy.stats

# Internal modules


logger = logging.getLogger(__name__)


def random_walk(params, noise=0.1, **kwargs):
    """
    Given parameters are disturbed by truncated Gaussian noise with given
    relative standard deviation. Truncation leads to unbiased and positive
    parameter values. Random walk is a stochastic process, where only the
    ensemble uncertainty is increased with time, while the expected value
    remains the same.

    Parameters
    ----------
    params : :py:class:`numpy.ndarray`
        This array of parameters is propagated in time by random walk.
    noise : float, optional
        This noise level (default = 0.1) is the standard deviation of the
        truncated Gaussian noise, relative to the parameter values. This noise
        level has to be between 0 and 1.
    kwargs : dict
        These additional keyword arguments are not used for this propagation
        model.

    Returns
    -------
    new_params : :py:class:`numpy.ndarray`
        These new parameters are given parameters added to the random walk with
        specified noise level.
    """
    scale = params * noise
    trunc_norm = scipy.stats.truncnorm(-1/noise, 1/noise, scale=scale)
    delta_params = trunc_norm.rvs(size=params.shape)
    new_params = params + delta_params
    return new_params
