#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging

# internal modules
from pylawr.transform.transformer import Transformer
from pylawr.transform.memorymixin import MemoryMixin
from pylawr.utilities import log_decorator
from pylawr.field import tag_array, get_verified_grid
from pylawr.grid.cartesian import CartesianGrid

# external modules
import xarray as xr
import numpy as np
from skimage.feature import match_template

logger = logging.getLogger(__name__)

TAG_EXTRAPOLATION = "extrapolated"
"""
Tag to indicate that an extrapolation was applied
"""


class Extrapolator(Transformer, MemoryMixin):
    """
    This class can be used to extrapolate a field based on two previous fields
    using template matching. The template matching finds similar areas between
    two fields. Based on the distance between similar pixels a vector of pixel
    movement between to time steps is calculated. With the vector the current
    field can be shifted to a field of a following time step.

    Attributes
    ----------
    vector : tuple, float
        fitted movement vector in meters per second
    time : ``numpy.datetime64``
        timestep of fitted vector
    cut_percentage : float
        Offset defines cut of range for old data.
    correlation_threshold : float
        Minimal threshold of maximal correlation of match template between
        data arrays. Is the correlation lower than threshold, the vector of
        movement is set to zero.
    max_timediff : int
        Maximal difference between time steps until warning.
    """

    def __init__(self, cut_percentage=0.15, correlation_threshold=0.5,
                 max_timediff=60.*10):
        super().__init__()
        self._time = None
        self._vector = None
        self.cut_percentage = cut_percentage
        self.correlation_threshold = correlation_threshold
        self.max_timediff = max_timediff
        self.direction = ['y', 'x']

        self.time = None
        self.vector = None

        self._trainable_vars = ('vector', 'time')

    @property
    def fitted(self):
        return self.time is not None

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, new_time):
        """
        ... sets timestep of fitted vector with new time. If `new_time` is
        `None` `time` resets on default.

        Parameters
        ----------
        new_time: ``numpy.datetime64`` or None
            new timestep of fitted vector
        """
        if new_time is None:
            self._time = new_time
        else:
            self._time = np.asarray(new_time, dtype="datetime64[ns]")

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, new_vector):
        """
        ... sets vector with new vector. If `new_vector` is
        `None` `vector` resets on default.

        Parameters
        ----------
        new_vector: tuple, int or None
            new fitted movement vector
        """
        self._vector = new_vector

    def _determine_timedelta(self, time, time_pre):
        """
        Get the timedelta between two data arrays.

        Parameters
        ----------
        time : ``numpy.datetime64``
            actual timestep.
        time_pre : ``numpy.datetime64``
            previous timestep.

        Returns
        -------
        timedelta : ``numpy.timedelat64`` (seconds)
            Timedelta in seconds.

        """
        timedelta = (time
                     - time_pre
                     ) / np.timedelta64(1, 's')

        logger.debug("Timedelta is {0:f} s".format(timedelta))

        if timedelta > self.max_timediff:
            logger.warning("Timedelta exceeds {0:f} minutes".format(
                self.max_timediff / 60.))

        return timedelta

    def calc_match_matrix(self, array, array_pre):
        """
        Calculates the correlation matrix based on match template between two
        arrays

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            Data of analogous actual timestep.
        array_pre : :py:class:`xarray.DataArray`
            Data of analogous previous timestep.

        Returns
        -------
        :py:class:`numpy.ndarray` (float)
            Correlation matrix

        """
        # cut_values
        cv = [int(round(self.cut_percentage * array_pre.values[0].shape[0])),
              int(round(self.cut_percentage * array_pre.values[0].shape[1]))]

        # match template based on correlation, nan must be replaced
        match_mat = match_template(np.nan_to_num(array.values[0]),
                                   np.nan_to_num(array_pre.values[0][
                                                 cv[0]:-cv[0],
                                                 cv[1]:-cv[1]]
                                                 )
                                   )

        return match_mat

    @staticmethod
    def _check_grids(grid_now, grid_pre):
        """
        Checks if the grids of the arrays are equal and
        child of :py:class:`pylawr.grid.cartesian.CartesianGrid`

        Parameters
        ----------
        grid_now : child of :py:class:`~pylawr.grid.base.BaseGrid`
            grid of data of actual timestep
        grid_pre : child of :py:class:`~pylawr.grid.base.BaseGrid`
            grid of data of previous timestep
        """
        if not isinstance(grid_now, CartesianGrid):
            raise TypeError(
                'Grid of actual timestep is not a cartesian grid.'
            )

        if not isinstance(grid_pre, CartesianGrid):
            raise TypeError(
                'Grid of previous timestep is not a cartesian grid.'
            )

        if not grid_now == grid_pre:
            raise ValueError(
                'The grids have different attributes.'
            )

    @log_decorator(logger)
    def fit(self, array, array_pre, grid=None, *args, **kwargs):
        """
        Fits a vector of movement between to arrays, of e.g. reflectivity,
        based on match template.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            Data of analogous actual timestep.
        array_pre : :py:class:`xarray.DataArray`
            Data of analogous previous timestep.
        grid : :py:class:`pylawr.grid.cartesian.CartesianGrid` or None, optional
            To fit the arrays they need to have the same grid specifications. If
            no grid is given, the grids from given arrays are used.
        """
        if grid is None:
            grid = get_verified_grid(array)
            self._check_grids(grid, get_verified_grid(array_pre))
        elif not isinstance(grid, CartesianGrid):
            raise TypeError(
                'Given grid is not a CartesianGrid'
            )
        else:
            array.lawr.check_grid(grid)
            array_pre.lawr.check_grid(grid)

        match_mat = self.calc_match_matrix(array, array_pre)

        # It was assumed
        # (... if the max_correlation is < 0.3 there is no rain)
        # ... if the max_correlation is < 0.5 the speed is set to zero
        if np.max(match_mat) >= self.correlation_threshold:
            # calculate the movement based on the match-result
            ij = np.unravel_index(np.argmax(match_mat),
                                  match_mat.shape)
            x, y = ij[::-1]
            dist_x = np.around((x - np.array(match_mat.shape[1]) * .5 + .5)
                               ).astype(int)
            dist_y = np.around((y - np.array(match_mat.shape[0]) * .5 + .5)
                               ).astype(int)
            dist = np.array([dist_y, dist_x])

            timedelta = self._determine_timedelta(self._get_time(array),
                                                  self._get_time(array_pre))

            if timedelta == 0.:
                logger.debug(
                    "The timedelta is zero. You fit the extrapolator to "
                    "the same timesteps".format(self.correlation_threshold)
                )
                self.vector = np.array([0, 0])
            else:
                self.vector = grid.resolution * dist / timedelta
        else:
            logger.debug(
                "The correlation is under {0:.2f}. The speed is set to "
                "zero!".format(self.correlation_threshold)
            )
            self.vector = np.array([0, 0])

        logger.info('Fitted extrapolator with movement vector {0:f} m in x '
                    'and {1:f} m in y'.format(self.vector[1],
                                              self.vector[0]))

        self.time = array.time.values

    @log_decorator(logger)
    def transform(self, array, grid=None, time=None,
                  *args, **kwargs):
        """
        Extrapolate the given given ``array`` with a cartesian ``grid``
        according to the fitted movement vector.

        Parameters
        ----------
        array: :py:class:`xarray.DataArray`
            The array to operate on
        grid : child of :py:class:`pylawr.grid.base.BaseGrid` or None
            To extrapolate the array needs to have a
            :py:class:`pylawr.grid.cartesian.CartesianGrid`.
        time : ``numpy.datetime64``
            next time step to extrapolate to
        args: sequence
            Further positional arguments
        kwargs: dict
            Further keyword arguments

        Returns
        -------
        :py:class:`xarray.DataArray`
            The transformed array
        """
        if grid is None:
            grid = get_verified_grid(array)

        timedelta_new = self._determine_timedelta(time,
                                                  self._get_time(array))

        pixel_move = np.around(self.vector
                               / grid.resolution
                               * timedelta_new).astype(int)

        # expand field with nans and roll
        field_new = np.pad(array.values,
                           ((0, 0),
                            (abs(pixel_move[0]), abs(pixel_move[0])),
                            (abs(pixel_move[1]), abs(pixel_move[1]))),
                           'constant', constant_values=np.nan)
        field_new = np.roll(field_new, pixel_move[0], axis=1)
        field_new = np.roll(field_new, pixel_move[1], axis=2)

        y_slice = slice(
            abs(pixel_move[0]), (field_new.shape[-2]-abs(pixel_move[0]))
        )
        x_slice = slice(
            abs(pixel_move[1]), (field_new.shape[-1]-abs(pixel_move[1]))
        )
        field_new = field_new[..., y_slice, x_slice]

        # transform new array
        transformed = array.copy()
        transformed.values = field_new
        transformed['time'] = [time]
        transformed = transformed.lawr.set_grid_coordinates(grid)
        tag_array(transformed, TAG_EXTRAPOLATION)

        logger.info('Extrapolated reflectivity - '
                    'moved pixels '
                    '{0:d} in x and {1:d} in y'.format(pixel_move[1],
                                                       pixel_move[0]))

        return transformed

    def to_xarray(self):
        """
        Serialize this filter's parameters to an :py:class:`xarray.Dataset`

        Returns
        -------
        :py:class:`xarray.Dataset`
            the filter's parameters as dataset
        """
        data_vars = {
            'cut_percentage': {
                'dims': (),
                'data': self.cut_percentage,
                'attrs': {}
            },
            'correlation_threshold': {
                'dims': (),
                'data': self.correlation_threshold,
                'attrs': {}
            },
            'max_timediff': {
                'dims': (),
                'data': self.max_timediff,
                'attrs': {'unit': 's'}
            },
            'time': {
                'dims': ('time', ),
                'data': self.time,
                'attrs': {}
            },
            'direction': {
                'dims': ('direction', ),
                'data': self.direction,
                'attrs': {}
            },
            'vector': {
                'dims': ('direction', ),
                'data': self.vector,
                'attrs': {
                    'unit': 'm/s',
                    'long_name': 'Movement vector'
                }
            }
        }

        ds = xr.Dataset.from_dict(data_vars)
        ds.attrs["type"] = self.__class__.__name__

        return ds
