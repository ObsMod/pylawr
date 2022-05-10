#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging

# external modules
import numpy as np

# internal modules
from pylawr.transform.filter.clutterfilter import ClutterFilter
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.spatial.beamexpansion import BeamExpansion, \
    TAG_BEAM_EXPANSION_CORR
from pylawr.field import array_has_tag

logger = logging.getLogger(__name__)


class SNR(ClutterFilter):
    """
    Calculates the Signal-to-Noise ratio.

    Attributes
    ----------
    threshold: float
            The threshold of the reflectivities to filter (linear reflectivity)
    """
    def __init__(self, threshold=0):
        super().__init__(threshold)
        self._threshold = threshold
        self.map = None

    def calc_map(self, array, noise_remover):
        """
        Computes Signal-to-Noise ratio. The array should not corrected for
        beam expansion.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
                The reflectivity array to operate on.
        noise_remover : ``NoiseRemover`` object
                Actual ``NoiseRemover`` which operates on RadarField
        """
        # make sure given array is on linear scale
        array = array.lawr.to_z()

        # make sure given array has uncorrected beam expansion
        if array_has_tag(array, TAG_BEAM_EXPANSION_CORR):
            beam_expansion = BeamExpansion()
            array = beam_expansion.transform(array)

        noise_old = noise_remover.noiselevel

        self.map = array.values / noise_old

    def create_cluttermap(self, array=None, noise_remover=None):
        """
        Creates clutter map based on the field and threshold.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
               The reflectivity array to operate on.
        noise_remover : :py:class:`~pylawr.transform.temporal.NoiseRemover`
                Actual ``NoiseRemover`` which operates on RadarField

        Returns
        -------
        cmap :  :py:class:`~pylawr.transform.filter.ClutterMap`
                The Cluttermap of ``_map`` to transform the reflectivity,
                contains probabilities of clutter (between one and zero).
        """
        if array is not None and noise_remover is not None:
            self.calc_map(array, noise_remover)

        clutter = np.zeros(self.map.shape)
        clutter[self.map >= self._threshold] = 1

        return ClutterMap('SNR', clutter)
