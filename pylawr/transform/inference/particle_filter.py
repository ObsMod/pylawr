#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np
import pandas as pd
import xarray as xr

# Internal modules
from ..memorymixin import MemoryMixin


logger = logging.getLogger(__name__)


class SIRParticleFilter(MemoryMixin):
    """
    This sequential importance resampling (SIR) particle filter can be used
    to infer unknown parameters and states with known observations and models.

    The particle filter needs an observation operator ``obs_op_func``, which
    maps from state space into observation space. An additional dynamical
    model ``predict_func`` is needed to propagate the state from one time
    step to the other. The probability function ``prob_func`` specifies the
    probability distribution of the observation space. The first guess
    parameters ``params_fg`` are used as first forecast of the parameters.

    Parameters
    ----------
    obs_op_func : function
        This func takes the state space parameters and transfers these into
        observation space. There are some prepared observation operators
        specified in this submodule.
    predict_func : function
        This prediction function is used as dynamical model to propagate the
        state space parameters from one time to another. This prediction
        function takes state space parameters and returns the predicted new
        state space parameters. There are some prepared prediction function
        specified in this submodule.
    prob_func : function
        This probability function specifies the probability density function
        of the observations. This probability takes observations generated
        with given observation operator and compares them to real
        observations. There are some commonly used probability density
        functions specified within this submodule.
    params_fg : iterable
        This parameters are used as first guess. The number of different
        parameters have to be equal the number of ensemble members. This is
        normally a :py:class:`numpy.ndarray`, where the first dimension is
        the ensemble size and the second dimension is the `i`-th parameter.
    ens_size : int, optional
        This number of ensemble members (default=100) is used to infer the
        parameters.
    ens_threshold : int, optional
        If the effective number of ensemble members is smaller than this
        threshold (default=10), the ensemble is resampled.
    """
    def __init__(self, obs_op_func, predict_func, prob_func,
                 params_fg, ens_size=100, ens_threshold=10,):
        self.obs_op_func = obs_op_func
        self.predict_func = predict_func
        self.prob_func = prob_func
        self.ens_threshold = ens_threshold
        self._ens_size = ens_size
        self._param_hist = [params_fg]
        self._weight_hist = [np.ones(ens_size)/ens_size]
        self._times = [pd.NaT]

    @property
    def ens_size(self):
        """
        The ensemble size of this particle filter.

        Returns
        -------
        ens_size : int
            The ensemble size of this particle filter.
        """
        return self._ens_size

    @property
    def param_hist(self):
        """
        The history of the parameters.

        Returns
        -------
        param_hist : :py:class:`numpy.ndarray`
            This is the parameter history, where the first axis is given as
            number of iterations. Normally, this array has as shape `n x k x p`
            with `n` the number of historic iterations, `k` the number of
            ensemble members and `p` as number of parameters.
        """
        return np.array(self._param_hist)

    @property
    def weight_hist(self):
        """
        The history of the estimated ensemble weights.

        Returns
        -------
        weight_hist : :py:class:`numpy.ndarray`
            This is the weight history with `n x k` as shape, `n` the number of
            historic iterations and `k` the number of ensemble members. At every
            iteration the weights are normalized to unity.
        """
        return np.array(self._weight_hist)

    @property
    def fitted(self):
        """
        If this particle filter is fitted.

        Returns
        -------
        fitted : bool
            If there was at least one iteration with this particle filter.
        """
        return len(self.param_hist) > 1

    def _propagate(self, **kwargs):
        return self.predict_func(self.param_hist[-1], **kwargs)

    def _update_weights(self, params, obs, **kwargs):
        new_weights = []
        for ens_mem in range(self.ens_size):
            tmp_obs_hat = self.obs_op_func(params[ens_mem], obs=obs, **kwargs)
            tmp_nll = self.prob_func(tmp_obs_hat, obs, **kwargs)
            tmp_weight = np.log(self._weight_hist[-1][ens_mem] + 1e-10) + tmp_nll
            new_weights.append(tmp_weight)
        new_weights = np.array(new_weights)
        new_weights = new_weights - np.max(new_weights)
        new_weights = np.exp(new_weights)
        new_weights = new_weights / np.sum(new_weights)
        return new_weights

    def _resample(self, params, weights, rnd=None, **kwargs):
        if isinstance(rnd, int) or rnd is None:
            rnd = np.random.RandomState(rnd)
        idx = rnd.choice(self.ens_size, size=self.ens_size, p=weights)
        new_params = params[idx]
        new_weights = np.ones_like(weights) / self.ens_size
        return new_params, new_weights

    def fit(self, obs, time=None, **kwargs):
        """
        This method iterates the particle filter with given observations. If the
        number of effective particles is below a set threshold, the ensemble
        members are resampled.

        Parameters
        ----------
        obs : :py:class:`numpy.ndarray`
            The conditional probability of the parameters with respect to these
            observations are used to update the particle weights.
        time : :py:class:`pandas.DateTime` or None, optional
            This time is used as timestamp within the weight and parameter
            history. If no time is given (default), an empty time step is used.
        **kwargs : dict(str, any)
            These additional keyword arguments are passed to the observation
            operator and the prediction function.
        """
        new_params = self._propagate(**kwargs)
        new_weights = self._update_weights(new_params, obs, **kwargs)
        eff_particles = 1 / np.sum(new_weights ** 2)
        if eff_particles < self.ens_threshold:
            new_params, new_weights = self._resample(new_params, new_weights,
                                                     **kwargs)
        self._param_hist.append(new_params)
        self._weight_hist.append(new_weights)
        fit_params = np.average(new_params, axis=0, weights=new_weights)
        if time is None:
            self._times.append(pd.NaT)
        else:
            self._times.append(pd.to_datetime(time))
        logger.info(
            'Fitted parameters with particle filter – n_eff: {0:.1f}, '
            'parameters: {1:s}'.format(eff_particles, str(fit_params))
        )
        return fit_params

    def to_xarray(self):
        parameters = xr.DataArray(
            data=np.array(self.param_hist),
            coords={
                'time': self._times,
                'ensemble': range(self.ens_size)
            },
            dims=['time', 'ensemble', 'param_id']
        )
        weights = xr.DataArray(
            data=np.array(self.weight_hist),
            coords={
                'time': self._times,
                'ensemble': range(self.ens_size)
            },
            dims=['time', 'ensemble']

        )
        ds = xr.Dataset({
            'parameters': parameters,
            'weights': weights
        })
        ds.attrs["type"] = self.__class__.__name__
        return ds
