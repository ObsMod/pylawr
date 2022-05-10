#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Internal modules
from pylawr.utilities import log_decorator
from pylawr.transform.memorymixin import MemoryMixin
from pylawr.grid import PolarGrid

logger = logging.getLogger(__name__)


class AttenuationCorrectionDual(MemoryMixin):
    """
    This class provides methods to
    estimate the attenuation based on the maximum apparent attenuation comparing
    reflectivities of attenuation-influenced frequency bands (X-band) and
    observations from less attenuated radar systems (C-band). The reflectivites
    have to be on the same grid. One approach is to interpolate the
    reflectivites on a joint coarse grid, estimate the attenuation and
    interpolate the attenuation on the fine grid of the attenuation-influenced
    frequency band.

    To estimate the attenuation of the attenuation-influenced frequency band
    (X-band) with the less attenuated radar system (C-band) the maximum apparent
    attenuation :math:`K_{\mathrm{max}}` (dB) is used:

    .. math:
        K_{\mathrm{max}} &= 10\cdot{}log[z_{\mathrm{C}} / z_{\mathrm{X}}] \\
                         &= Z_{\mathrm{C}} - Z_{\mathrm{X}}

    The attenuation should increase with increasing distance in theory. To get
    an increasing attenuation some regression is applied on
    :math:`K_{\mathrm{max}}`, e.g. the isotonic regression. For further
    information look up Lengfeld et. al (2016).

    Attributes
    ----------
    attenuation : :py:class:`xarray.DataArray` or None
        Estimated attenuation for the correction, negative values are not
        allowed
    """

    def __init__(self):
        super().__init__()
        self._attenuation = None
        self._trainable_vars = ('attenuation',)
        self.attenuation = None

    @property
    def attenuation(self):
        return self._attenuation

    @attenuation.setter
    def attenuation(self, new_attenuation):
        """
        ... sets values for the attenuation correction.

        Parameters
        ----------
        new_attenuation: :py:class:`xarray.DataArray` or None
            new attenuation field
        """
        self._attenuation = new_attenuation

    @property
    def fitted(self):
        return self.attenuation is not None

    @staticmethod
    def _check_grids(grid_first, grid_second):
        """
        Checks if the grids are child of :py:class:`pylawr.grid.polar.PolarGrid`
        and have equal attributes.

        Parameters
        ----------
        grid : child of :py:class:`pylawr.grid.base.BaseGrid`
            grid of data
        """
        if not isinstance(grid_first, PolarGrid):
            raise TypeError(
                'First grid is not a polar grid.'
            )

        if not isinstance(grid_second, PolarGrid):
            raise TypeError(
                'Second grid is not a polar grid.'
            )

        if not grid_first == grid_second:
            raise ValueError(
                'The grids have different attributes.'
            )

    @staticmethod
    def _calc_kmax(refl_attenuated, refl_robust):
        """
        Calculate the attenuation factor between attenuated reflectivity and
        robust reflectivity. Only where the attenuated reflectivity is available
        the attenuation factor is calculated (otherwise we would create a
        reflectivity signal due to e.g. different radar resolution and
        incorrect alignment).

        Parameters
        ----------
        refl_attenuated: :py:class:`xarray.DataArray`
            Reflectivity of attenuation-influenced radar system
        refl_robust: :py:class:`xarray.DataArray`
            Reflectivity of less attenuated radar system

        Returns
        -------
        k_max : :py:class:`xarray.DataArray`
            The attenuation between the two radar fields.
        """
        k_max = refl_robust - refl_attenuated
        k_max.values[refl_attenuated.values < 0] = np.nan
        return k_max

    @staticmethod
    def _regress_attenuation(k_max,
                             regression=IsotonicRegression(),
                             replace_neg=0):
        """
        Calculate attenuation based on given regression object.

        Parameters
        ----------
        k_max : :py:class:`xarray.DataArray`
            The attenuation is estimated based on this array.
        regression: object
            Regression object with `.fit_transform(x, y)` method. Default is
            :py:class:`sklearn.isotonic.IsotonicRegression`.
        replace_neg: float, NaN or None
            Negative values are replace with float values or `np.nan`. If this
            parameter is set to `None` negative values will not replaced, e.g.
            for research reasons.

        Returns
        -------
        attenuation : :py:class:`xarray.DataArray`
            Estimated attenuation array based on given reflectivity differences
            and regression.
        """
        attenuation = k_max.interpolate_na(dim='range')
        attenuation = k_max.fillna(0.)
        for ind in np.ndindex(*k_max.shape[:-1]):
            attenuation[ind] = regression.fit_transform(
                attenuation['range'].values.astype(np.float64), attenuation[
                    ind]
            )
        if replace_neg is not None:
            attenuation = attenuation.where(attenuation > 0, replace_neg)
        attenuation = attenuation.where(~np.isnan(k_max))
        return attenuation

    @log_decorator(logger)
    def fit(self, refl_attenuated, refl_robust,
            regression=IsotonicRegression(),
            replace_neg=0):
        """
        Estimates the attenuation correction. Negative values are not allowed,
        otherwise grid resolution effects of the less attenuated radar systems
        could occure. In default the isotonic regression is applied according
        to Lengfeld et. al (2016).

        Parameters
        ----------
        refl_attenuated: :py:class:`xarray.DataArray`
            Reflectivity of attenuation-influenced radar system
        refl_robust: :py:class:`xarray.DataArray`
            Reflectivity of less attenuated radar system
        regression: object
            Regression object with `.fit_transform(x, y)` method. Default is
            :py:class:`sklearn.isotonic.IsotonicRegression`. If this is None,
            no regression is used and raw differences between given fields are
            used as attenuation.
        replace_neg: float, NaN or None
            Negative values are replace with float values or `np.nan`. If this
            parameter is set to `None` negative values will not replaced, e.g.
            for research reasons.
        """
        self._check_grids(refl_attenuated.lawr.grid,
                          refl_robust.lawr.grid)

        k_max = self._calc_kmax(refl_attenuated, refl_robust)
        if regression is None:
            attenuation = k_max
        else:
            attenuation = self._regress_attenuation(
                k_max, regression=regression, replace_neg=replace_neg
            )

        self.attenuation = attenuation
        self.attenuation = self.attenuation.lawr.set_variable('pia')

    def to_xarray(self):
        """
        Serialize this filter's parameters to an :any:`xarray.Dataset`

        Returns
        -------
        :any:`xarray.Dataset`
            the filter's parameters as dataset
        """
        ds = self.attenuation.to_dataset()
        ds.attrs["type"] = self.__class__.__name__

        return ds
