#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging
import abc

logger = logging.getLogger(__name__)


class Transformer(abc.ABC):
    """
    Abstract base class for transformers, which apply filters or other
    operations on `RadarField`
    """

    @abc.abstractmethod
    def transform(self, array, grid=None, *args, **kwargs):
        """
        Transform a given ``array`` with an optional ``grid`` according to the
        parameters.

        Parameters
        ----------
        array: :py:class:`xarray.DataArray`
            The array to operate on.
        grid : child of :py:class:`pylawr.grid.base.BaseGrid`, optional
            The grid to use. If left unspecified,
            ``array.lawr.grid`` is used.
        args: sequence
            Further positional arguments
        kwargs: dict
            Further keyword arguments

        Returns
        -------
        :py:class:`xarray.DataArray`
            The transformed array
        """
        pass
