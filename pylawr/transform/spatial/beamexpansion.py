#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging

# external modules
import xarray

# internal modules
from pylawr.transform.transformer import Transformer
from pylawr.field import tag_array, array_has_tag, untag_array, \
    get_verified_grid
from pylawr.utilities.decorators import log_decorator

logger = logging.getLogger(__name__)

TAG_BEAM_EXPANSION_CORR = "beam-expansion-corr"
TAG_BEAM_EXPANSION_UNCORR = "beam-expansion-uncorr"


class BeamExpansion(Transformer):
    r"""
    Filter to apply/remove the beam expansion effect to/from a reflectivity
    field. See `radartutorial.eu <http://radartutorial.eu>`_ for reference.

    This filter basically just does

    .. math::

        input \cdot R ^ 2

    in the forward direction and

    .. math::

        \frac{input}{R ^ 2}

    in the inverse direction.

    """

    @staticmethod
    def _determine_inv(array):
        """
        Method to determine the transform direction if no inverse is given.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            The **linear** reflectivity array to operate on.

        Returns
        -------
        inverse : bool
            The direction of the transformation with apply if False or removed
            if True.
        """
        inverse = False
        corr = array_has_tag(array, TAG_BEAM_EXPANSION_CORR)
        uncorr = array_has_tag(array, TAG_BEAM_EXPANSION_UNCORR)
        if corr and uncorr:
            raise ValueError(
                "Array is tagged both with '{}' and '{}' "
                "which does not make sense. Please specify "
                "'inverse=True/False'.".format(
                    TAG_BEAM_EXPANSION_CORR, TAG_BEAM_EXPANSION_UNCORR
                )
            )
        elif corr:  # array is already corrected -> invert
            inverse = True
        return inverse

    @log_decorator(logger)
    def transform(self, array, grid=None, inverse=None, *args, **kwargs):
        """
        Apply (or remove if ``inverse=True``) the beam expansion effect from a
        given ``array``.

        Tag the array with
        :py:const:`~pylawr.transform.spatial.beamexpansion.TAG_BEAM_EXPANSION_CORR`
        or
        :py:const:`~pylawr.transform.spatial.beamexpansion.TAG_BEAM_EXPANSION_UNCORR`
        accordingly.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            The **linear** reflectivity array to operate on.
        grid : child of :py:class:`pylawr.grid.base.BaseGrid` or None
            This grid is used to determine the distance to the radar. If no
            grid is given it will be determined from the array.
        inverse : bool or None, optional
            Invert the transformation? Defaults to ``None``, which means to try
            to determine the direction automatically by checking the array's
            :py:attr:`~pylawr.RadarField.tags`.
            If that fails, ``inverse`` defaults to ``False``.

        Returns
        -------
        transformed_array : :py:class:`xarray.DataArray`
            The array with the beam expansion effect applied/removed

        Raises
        ------
        AttributeError
            An AttributeError is raised if no grid was specified and no grid is
            set for the array.
        TypeError
            A TypeError is raised if the given grid is not a valid grid.
        ValueError
            A ValueError is raised if the given grid does not have the same
            shape as the given array.
        """
        grid = get_verified_grid(array, grid=grid)
        if inverse is None:
            inverse = self._determine_inv(array)
        if inverse:
            transformed_array = array / grid.center_distance ** 2
            transformed_array = transformed_array.lawr.set_metadata(array)
            untag_array(transformed_array, TAG_BEAM_EXPANSION_CORR)
            tag_array(transformed_array, TAG_BEAM_EXPANSION_UNCORR)
        else:
            transformed_array = array * grid.center_distance ** 2
            transformed_array = transformed_array.lawr.set_metadata(array)
            untag_array(transformed_array, TAG_BEAM_EXPANSION_UNCORR)
            tag_array(transformed_array, TAG_BEAM_EXPANSION_CORR)

        return transformed_array

