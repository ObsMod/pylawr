#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import warnings

# External modules
import numpy as np
import scipy.ndimage

# Internal modules
from pylawr.functions.grid import get_masked_grid
from pylawr.remap.nearest import NearestNeighbor
from pylawr.field import tag_array, get_verified_grid
from pylawr.transform.transformer import Transformer
from pylawr.utilities.helpers import polar_padding


logger = logging.getLogger(__name__)


class Interpolator(Transformer):
    """
    This class can be used to fill holes within a radar field with given
    remapping algorithm. For the interpolation temporary
    :py:class:`~pylawr.grid.UnstructuredGrid` is used.

    Parameters
    ----------
    threshold : float
        If more values are nan than field size * threshold, then no
        interpolation is possible. Default is 0.75.
    algorithm : child of :py:class:`pylawr.remap.base.BaseRemap` or None
        This initialized algorithm is used for hole filling. If this is
        None, :py:class:`~pylawr.remap.NearestNeighbor` with one
        nearest neighbor is used as algorithm. Default is None.
    polar : boolean, optional
        If the given reflectivity is in polar coordinates or in other
        coordinates. If polar coordinates are used, the padding for the masks
        is calculated with polar padding, else a reflect padding is used.
    cov_thres : float, optional
        If this interpolation covariance (uncertainty) above this threshold (
        default :math:`= 4~\text{dbz}^{2}), then the value is not interpolated.
        If given interpolation algorithm has no uncertainty estimation, this
        step is skipped. To use no thresholding this values can be set to
        :py:class:`numpy.inf`.
    zero_field : tuple(int)
        The size of the local receptive field to search for valid values, which
        have rain. If the number of rain values is below a given threshold, the
        missing values are set to zero. Default is (11, 11).
    zero_thres : float, optional
        If the number of rain values within a local receptive field is below
        this threshold, the missing values are set to zero. Default is 0.34.

    Warnings
    --------
    The algorithm is refitted every time such that old fits are overwritten.
    """
    def __init__(self, threshold=0.75, algorithm=None, polar=True, cov_thres=4,
                 zero_field=(11, 11), zero_thres=0.34):
        self.threshold = threshold
        self._algorithm = None
        self.algorithm = algorithm
        self.polar = polar
        self.cov_thres = cov_thres
        self.zero_field = zero_field
        self.zero_thres = zero_thres

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, new_algorithm):
        if new_algorithm is None:
            self._algorithm = NearestNeighbor(n_neighbors=1)
        elif hasattr(new_algorithm, 'remap'):
            self._algorithm = new_algorithm
        else:
            raise TypeError('The given algorithm is not None or a valid '
                            'remapping algorithm')

    def _get_source_mask(self, reflectivity):
        """
        Get a mask for all source points based on given reflectivity. Two
        different conditions are necessary such that a value is declared as
        source value:

            - the value is not nan
            - the value has no surrounding nan value or is below 5 dBZ

        Parameters
        ----------
        reflectivity : :py:class:`xarray.DataArray`
            This reflectivity array is used to create the source mask.

        Returns
        -------
        source_mask : :py:class:`numpy.ndarray` (bool)
            All source values are True, all other are False.
        """
        with np.errstate(invalid='ignore'):
            rain_pixels = reflectivity.lawr.get_rain_mask()
        non_missing = np.isfinite(reflectivity.values)
        filter_size = [1, ]*(reflectivity.ndim-2) + [3, 3]
        if self.polar:
            padded_non_missing = polar_padding(non_missing, pad_size=(1, 1))
            non_missing_surround_mask = scipy.ndimage.minimum_filter(
                padded_non_missing, size=filter_size)[..., 1:-1, 1:-1]

            padded_rain_pixels = polar_padding(rain_pixels, pad_size=(1, 1))
            mean_rain_pixels = scipy.ndimage.uniform_filter(
                padded_rain_pixels.astype('float'), size=filter_size
            )[..., 1:-1, 1:-1]
        else:
            non_missing_surround_mask = scipy.ndimage.minimum_filter(
                non_missing, size=filter_size, mode='reflect')
            mean_rain_pixels = scipy.ndimage.uniform_filter(
                rain_pixels.astype('float'), size=filter_size, mode='reflect'
            )

        num_rain_pixels = mean_rain_pixels * 9
        rain_pixels_submask = num_rain_pixels > 4
        rain_pixels_mask = np.logical_and(rain_pixels, rain_pixels_submask)
        no_rain_mask = np.logical_and(non_missing, ~rain_pixels)
        criterion_mask = np.logical_or(
            np.logical_or(non_missing_surround_mask, no_rain_mask),
            rain_pixels_mask
        )
        source_mask = np.logical_and(
            non_missing,
            criterion_mask
        )
        return source_mask

    @staticmethod
    def _get_target_mask(reflectivity):
        """
        Mask all values, which should be interpolated. Currently, only
        :py:class:`numpy.nan` are set to True, while all other are set to False.

        Parameters
        ----------
        reflectivity : :py:class:`xarray.DataArray`
            This reflectivity array is used to create the target mask. All nan-
            values are used as target.

        Returns
        -------
        target_mask : :py:class:`numpy.ndarray` (bool)
            The inferred boolean target mask with the same shape as the
            given reflectivity.
        """
        target_mask = np.isnan(reflectivity.values)
        return target_mask

    def _get_valid_mask(self, reflectivity):
        """
        Get a mask for all valid values. A value is declared as valid value, if
        the number of rain values (> 5 dBZ) within a local receptive field is
        larger than a set threshold.

        Parameters
        ----------
        reflectivity : :py:class:`xarray.DataArray`
            This reflectivity array is used to create the valid mask.

        Returns
        -------
        valid_mask : :py:class:`numpy.ndarray` (bool)
            All valid values are True, all other are False.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rain_mask = (reflectivity.squeeze() > 5).astype(float)
        if self.polar:
            pad_size = tuple(((np.array(self.zero_field[-2:])-1)/2).astype(int))
            padded_rain_mask = polar_padding(rain_mask, pad_size=pad_size)
            valid_mean = scipy.ndimage.uniform_filter(
                padded_rain_mask, size=self.zero_field, mode='reflect'
            )[..., pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]
        else:
            valid_mean = scipy.ndimage.uniform_filter(
                rain_mask, size=self.zero_field, mode='reflect'
            )
        valid_mask = valid_mean > self.zero_thres
        return valid_mask

    def _prefill_noninterp_vals(self, reflectivity):
        """
        All missing and not valid values within a given array are set to
        -32.5 dBZ.

        Parameters
        ----------
        reflectivity : :py:class:`xarray.DataArray`
            This reflectivity array is used to fill non valid values

        Returns
        -------
        filled_values : :py:class:`xarray.DataArray`
            In this array all missing and not valid values are replaced with
            -32.5 dBZ.
        """
        filled_values = reflectivity.copy()
        target_mask = self._get_target_mask(reflectivity=reflectivity)
        valid_mask = self._get_valid_mask(reflectivity=reflectivity)
        zero_mask = np.logical_and(target_mask, ~valid_mask)
        filled_values.values[zero_mask] = -32.5
        return filled_values

    def _repl_uncertain_interpolation(self, refl_array, repl_value=np.nan):
        """
        This method replaces values in a given interpolation array, where the
        interpolation covariance are above a set threshold.
        """
        refl_replaced = refl_array.copy()
        if hasattr(self.algorithm, 'covariance'):
            cov_over_thres = self.algorithm.covariance > self.cov_thres
            logger.debug(
                'Interpolation values over covariance threshold: {0:d}'.format(
                    np.sum(cov_over_thres)
                )
            )
            refl_replaced[..., cov_over_thres] = repl_value
        return refl_replaced

    def transform(self, refl_array, grid=None, *args, **kwargs):
        """
        Interpolate holes within given array.

        Parameters
        ----------
        refl_array : :py:class:`xarray.DataArray`
            This array is used as input array. This array is handled as single
            reflectivity array such that only one time step at a time can be
            processed.
        grid : child of :py:class:`pylawr.grid.base.BaseGrid` or None
            This grid is used for the interpolation. If this grid is None, the
            grid from given array is used. Default is None.

        Returns
        -------
        filled_array : :py:class:`xarray.DataArray`
            In this interpolated array all nan-values are replaced by their
            interpolated value. The interpolated array will be returned as
            logarithmic reflectivity in dBZ.

        Warnings
        --------
        For interpolation of missing values a grid is needed.
        """
        refl_array = refl_array.lawr.to_dbz()
        nan_vals = np.sum(np.isnan(refl_array.values))
        logger.debug('Nan values before interpolation: {0:d}'.format(nan_vals))
        origin_grid = get_verified_grid(array=refl_array, grid=grid)

        source_mask = self._get_source_mask(reflectivity=refl_array).squeeze()
        source_grid = get_masked_grid(mask_array=source_mask,
                                      origin_grid=origin_grid)

        zero_filled_array = self._prefill_noninterp_vals(refl_array)
        stacked_data = zero_filled_array.stack(
            grid_cell=origin_grid.coord_names
        )
        source_data = stacked_data.copy()[..., :source_grid.grid_shape[0]]
        source_data[:] = refl_array.values[..., source_mask]

        target_mask = self._get_target_mask(
            reflectivity=zero_filled_array
        ).squeeze()
        target_grid = get_masked_grid(mask_array=target_mask,
                                      origin_grid=origin_grid)

        if target_grid.grid_shape == (0, ) or source_grid.grid_shape == (0, ):
            logger.warning('No / only NaN value(s) found within reflectivity, '
                           'no interpolation possible!')
            filled_array = zero_filled_array
        else:
            logger.info(
                'Before interpolation: {0:d} NaN values'.format(
                    target_grid.size
                )
            )
            self.algorithm.fit(grid_in=source_grid, grid_out=target_grid)
            int_data = self.algorithm.remap(data=source_data).values.squeeze()
            nan_vals = np.isnan(stacked_data.values)
            stacked_data.values[nan_vals] = int_data
            stacked_data.values[nan_vals] = self._repl_uncertain_interpolation(
                stacked_data.values[nan_vals]
            )
            filled_array = stacked_data.unstack('grid_cell')
            logger.info(
                'After interpolation: {0:d} NaN values'.format(
                    np.sum(np.isnan(filled_array.values))
                )
            )

        filled_array = filled_array.fillna(-32.5)
        filled_array = filled_array.lawr.set_metadata(refl_array)
        tag_array(filled_array,
                  'Interpolated with {0:s}'.format(str(self.algorithm)))
        return filled_array
