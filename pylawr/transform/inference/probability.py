#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


def gaussian_pdf(y_hat, y_obs, var=1, **kwargs):
    """
    This likelihood corresponds to a Gaussian probability distribution function.
    Given variance is directly used for the Gaussian PDF.

    Parameters
    ----------
    y_hat : :py:class:`numpy.ndarray`
        The predicted observations with the same shape as given observations.
    y_obs : :py:class:`numpy.ndarray`
        The actual observations, which are used as reference for the predicted
        observations.
    var : float or :py:class:`numpy.ndarray`, optional
        This is the variance for the Gaussian PDF. If this is a float
        (default=1), the variance is the same for all observations. To specify
        different uncertainties for different observations, one needs to give
        an array. This array should have the same size as given observations.
    kwargs : dict
        Additional keyword arguments which are not used in this function.

    Returns
    -------
    likeli : float
        This is the Gaussian negative log likelihood of the predicted
        observations for given actual observations, normed by the observational
        variance.
    """
    squared_res = (y_obs - y_hat) ** 2 / var / 2
    denom = np.sqrt(2 * np.pi * var)
    log_denom = np.log(denom + 1e-10)
    nll = np.nansum(-squared_res - log_denom)
    return nll


def laplace_pdf(y_hat, y_obs, var=1, **kwargs):
    """
    This likelihood corresponds to a Laplace probability distribution function.
    Given variance is automatically converted to the laplacian scaling
    coefficient.

    Parameters
    ----------
    y_hat : :py:class:`numpy.ndarray`
        The predicted observations with the same shape as given observations.
    y_obs : :py:class:`numpy.ndarray`
        The actual observations, which are used as reference for the predicted
        observations.
    var : float or :py:class:`numpy.ndarray`, optional
        This is the variance for the Laplace PDF. If this is a float
        (default=1), the variance is the same for all observations. To specify
        different uncertainties for different observations, one needs to give
        an array. This array should have the same size as given observations.
        This variance is automatically converted to the laplacian scaling
        coefficient.
    kwargs : dict
        Additional keyword arguments which are not used in this function.

    Returns
    -------
    likeli : float
        This is the Laplace negative log likelihood of the predicted
        observations for given actual observations, normed by the observational
        laplacian scaling.
    """
    scale = np.sqrt(var / 2)
    abs_res = np.abs(y_obs - y_hat) / scale
    denom = 2 * scale
    log_denom = np.log(denom + 1e-10)
    nll = np.nansum(-abs_res - log_denom)
    return nll

