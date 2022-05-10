#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import datetime
import logging
import inspect

# internal modules
from pylawr.transform.transformer import Transformer
from pylawr.transform.memorymixin import MemoryMixin
from pylawr.field import tag_array
from pylawr.utilities import log_decorator

# external modules
import xarray as xr
import numpy as np
from pandas import to_datetime

logger = logging.getLogger(__name__)

TAG_NEG_LIN_REFL_REPL = "neg-lin-refl-repl"
"""
Tag to indicate that negative linear reflectivities were replaced with
something.
"""

TAG_NOISE_FILTER = "noise-filtered"
"""
Tag to indicate that a noise filter was applied
"""


class NoiseRemover(Transformer, MemoryMixin):
    r"""
    Noise filter that dynamically determines an appropriate
    spatially independent noise threshold to subtract from the reflectivity
    field.

    **Assumptions:**

    - reflectivity data is the **raw measurement**, i.e. no range dependency
      correction etc. has happened yet

    Attributes
    ----------
    noise_percentile : float
        percentile of all sorted values to use for determination of a noise
        level. Defaults to ``10`` which means :math:`10\%`.
    remembrance_time : float
        time in seconds to do a moving average of older values. Defaults
        to 300 seconds, 5 minutes.
    max_noise_percentile_tendency : float
        maximum relative tendency per time of the actual :any:`noise_percentile`
        value. Defaults to :math:`\frac{20\%}{30s}`.
        If a higher relative tendency occurs, the previous
        :any:`noiselevel` will be used for filtering.
    max_time_constant_noiselevel : float
        maximum time in seconds where the :any:`noiselevel` can be constant
        before it is reset to the default. Defaults to three hours.
    noiselevel_constant_since : numpy.datetime64
        the time where :any:`noiselevel` last changed
    noiselevel_multiplier : float
        the noiselevel is multiplied with this constant value. Defaults to 1.03.
    times : numpy.ndarray(numpy.datetime64)
        recorded threshold times
    thresholds : numpy.ndarray
        recorded thresholds
    noiselevel : float
        current noise level
    """
    def __init__(
            self,
            noise_percentile=10,
            max_noise_percentile_tendency=0.2/30,
            max_time_constant_noiselevel=3*60*60,
            noiselevel_multiplier=1.03,
            remembrance_time=60*5,
            noiselevel=8.0e-7
            ):
        super().__init__()
        self.noise_percentile = noise_percentile
        self.max_noise_percentile_tendency = max_noise_percentile_tendency
        self.max_time_constant_noiselevel = max_time_constant_noiselevel
        self.noiselevel_multiplier = noiselevel_multiplier
        self.remembrance_time = remembrance_time

        self.default_noiselevel = noiselevel

        self.times = None
        self.thresholds = None

        self._trainable_vars = ('thresholds', 'times')

    # Properties
    @property
    def noiselevel(self):
        return self.get_noiselevel()

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, new_times):
        """
        ... sets recorded threshold times with new times. If `new_times` is
        `None` `times` resets on default.

        Parameters
        ----------
        new_times: numpy.ndarray(numpy.datetime64)
            new recorded threshold times
        """
        if new_times is None:
            self._times = np.array([], dtype="datetime64[ns]")
        else:
            self._times = np.asarray(new_times, dtype="datetime64[ns]")

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, new_thresholds):
        """
        ... sets recorded thresholds. If `new_thresholds` is `None` `thresholds`
        resets on default.

        Parameters
        ----------
        new_thresholds : numpy.ndarray(numpy.float)
            new recorded thresholds
        """
        if new_thresholds is None:
            self._thresholds = np.array([], dtype="float")
        else:
            self._thresholds = np.asarray(new_thresholds, dtype=float)

    @property
    def noiselevel_constant_since(self):
        """
        Get the index, where the noise levels changed the last time.

        Returns
        -------
        time_obj : numpy.datetime64 or numpy.nan
            The time of the last changed value within the thresholds. If the
            index is np.nan, then no noise level was saved or the noise level
            was saved without setting the appropriate time.
        """
        try:
            different = self.thresholds != self.thresholds[-1]
            reversed_different = different[::-1]
            ind = different.size - np.argmax(reversed_different)
            ind = min(ind, different.size-1)
            time_obj = self.times[ind]
        except IndexError:
            time_obj = datetime.datetime.now()
        logger.debug(
            'Determined {0:s} as last changed time'.format(str(time_obj))
        )
        return time_obj

    @property
    def xr_thresholds(self):
        """
        Get the thresholds as xarray.DataArray.

        Returns
        -------
        thres : xarray.DataArray
            The thresholds as xarray.DataArray with times as axis.
        """
        thres = xr.DataArray(
            data=self.thresholds,
            coords=dict(
                times=self.times
            ),
            dims=['times', ]
        )
        return thres

    @property
    def fitted(self):
        return self.times.size > 0

    # Methods
    def to_xarray(self):
        """
        Convert parameters to a dataset for easy serialisation

        Returns
        -------
        xarray.Dataset
            parameters as dataset
        """
        # get argument names to __init__ function
        var_keys = list(self._trainable_vars) + inspect.getfullargspec(
            self.__init__).args

        data_vars = {
            n: {} for n in var_keys
            if not n == "self"
        }

        # fill vars with data
        for var, d in data_vars.items():
            data_vars[var]["dims"] = ()
            data_vars[var]["data"] = getattr(self, var)
            data_vars[var]["attrs"] = {}

        # set units by hand
        data_vars["remembrance_time"]["attrs"]["unit"] = "s"
        data_vars["noise_percentile"]["attrs"]["unit"] = "%"
        data_vars["max_noise_percentile_tendency"]["attrs"]["unit"] = "1/s"
        data_vars["max_time_constant_noiselevel"]["attrs"]["unit"] = "s"
        data_vars["thresholds"]["attrs"]["unit"] = "Z"
        data_vars["noiselevel"]["attrs"]["unit"] = "Z"
        # set dimensions by hand
        data_vars["times"]["dims"] = ("times", )
        data_vars["thresholds"]["dims"] = ("times", )
        # convert to xarray.Dataset
        ds = xr.Dataset.from_dict(data_vars)
        ds.attrs["type"] = self.__class__.__name__
        return ds

    def _get_remembrance_interval(self, time_obj):
        """
        Calculate the remembrance interval based on given time object and set
        remembrance time.

        Parameters
        ----------
        time_obj : `datetime-like`
            The remembrance interval is calculated based on this time object.

        Returns
        -------
        start_date : pandas.datetime
            The start date of the remembrance interval. This is calculated
            based on :math:`time_obj-remembrance_time`
        end_date : pandas.datetime
            The end date of the remembrance interval. Basically, this is the
            given time object as pandas-datetime.
        """
        end_date = to_datetime(np.datetime64(time_obj))
        remembrance_delta = datetime.timedelta(
            seconds=int(self.remembrance_time)
        )
        start_date = end_date - remembrance_delta
        return start_date, end_date

    def _determine_noiselevel(self, time_obj):
        start_date, end_date = self._get_remembrance_interval(time_obj)
        noiselevels = self.xr_thresholds.sel(times=slice(start_date, end_date))
        noiselevel = self._get_mean_noise(noiselevels)
        return noiselevel

    @staticmethod
    def _get_mean_noise(noiselevels):
        """
        Calculate the mean noise level from given noise levels. At the moment
        this is only the nan median of the noise levels. In the future it will
        be possible to calculate the mean noise with a weighted mean.

        Parameters
        ----------
        noiselevels : :any: array-like
            The mean from these noise levels is calculated.

        Returns
        -------
        noise_level : float
            The calculated noise level.
        """
        return np.nanmedian(noiselevels)

    def get_noiselevel(self, time_obj=None):
        """
        Determine an appropriate noiselevel from the available data. If
        ``time`` is given, interpolate an appropriate noiselevel from
        :any:`thresholds`. If that is impossible, use the value of
        :any:`noiselevel`.

        Parameters
        ----------

        time_obj : datetime.datetime or numpy.datetime64, optional
            The time used to determine the noiselevel. If None the
            value of the last`noiselevel` is returned.

        Returns
        -------

        float
            the noiselevel to use
        """
        if not self.fitted:
            noiselevel = self.default_noiselevel
        elif time_obj is None:
            noiselevel = self._determine_noiselevel(self.times[-1])
        else:
            noiselevel = self._determine_noiselevel(time_obj)
        return float(noiselevel)

    @log_decorator(logger)
    def transform(self, array, time_obj=None, replace_negative=0.,
                  *args, **kwargs):
        """
        Subtract an appropriate noiselevel (:any:`get_noiselevel`) from the
        given **linear** reflectivity ``array``, optionally replacing
        (unphysical) negative
        results.

        Parameters
        ----------

        array : array-like
            The **linear** reflectivity array to remove the noise level from
        time_obj : datetime.datetime or numpy.datetime64, optional
            The time used to determine the noiselevel to use. Defaults to
            ``array.time`` if available. If unavailable, the value of
            :any:`noiselevel` is used for transforming.
        replace_negative : float, optional
            The value to replace negative values in the resulting array with.
            Defaults to :math:`0.\,\mathrm{mm^6m^{-3}}`.
            Set to :any:`None` to prevent replacing.

        Returns
        -------

        array-like
            The array with linear reflectivity and the noise level is removed.
        """
        if time_obj is None and hasattr(array, 'time'):  # no time given
            try:
                time_obj = array.time.values[0]
            except IndexError:
                time_obj = array.time.values

        noiselevel = self.get_noiselevel(time_obj=time_obj)
        logger.debug("Noiselevel is {0:f}".format(noiselevel))

        # make sure given array is on linear scale
        array = array.lawr.to_z()

        # do the actual "transforming" -> remove the noiselevel from the field
        transformed = array - self.noiselevel_multiplier * noiselevel

        # replace negative values if desired
        if replace_negative is not None:
            transformed = transformed.where(transformed > 0, replace_negative)
            tag_array(
                array,
                "{}-{}".format(TAG_NEG_LIN_REFL_REPL, str(replace_negative))
            )

        transformed = transformed.lawr.set_metadata(array)

        # add tag
        tag_array(transformed, TAG_NOISE_FILTER)

        return transformed

    def _is_rainy_field(self, array):
        """
        Check if RadarField has more rain than a threshold. The threshold is
        calculated as :math:`100 - noise_percentile`.

        Parameters
        ----------
        array : xarray.DataArray
            This array is used to check if the field is rainy or not. The array
            is converted into reflectivity and all reflectivity above 0 dBZ
            will be set to rain.

        Returns
        -------
        is_rainy : bool
            If the rain fraction is larger than the threshold, this will be
            True, else False.
        """
        rainy_threshold = 100-self.noise_percentile

        array_dbz = array.lawr.to_dbz()
        rain_array = array_dbz > 0
        rain_fraction = float(
            np.mean(
                rain_array
            )
        )
        logger.debug("{:.2f}% of the dataarray suggests rain".format(
            100*rain_fraction))
        logger.debug(
            "The 'a lot of rain'-threshold is {:.2f}".format(rainy_threshold)
        )
        is_rainy = rain_fraction * 100 > rainy_threshold
        return is_rainy

    def _get_old_noise_level(self, time_obj):
        """
        Get the noise level from the old thresholds, which is the nearest to
        the given time object.

        Parameters
        ----------
        time_obj : numpy.datetime64
            The nearest noise level to this time object will be returned.

        Returns
        -------
        old_noiselevel : float
            The old_noiselevel is either determined from nearest old noise level
            or if not fitted it is the default value.
        """
        if self.fitted:
            times_diff = self.times - time_obj
            older_than_given = times_diff.astype(float) < 0
            prev_ind = times_diff[older_than_given].argmax()
            old_noiselevel = self.thresholds[prev_ind]
        else:
            old_noiselevel = self.noiselevel
        return old_noiselevel

    def _determine_noiselevel_from_array(self, array):
        """
        Calculate the noise level from given array. At the moment, the noise
        level is the `noise_percentile`-th percentile of the given array.

        Parameters
        ----------
        array : xarray.DataArray
            This array is used to calculate the noise level.

        Returns
        -------
        noiselevel : float
            The calculated noise level, based on the given array.
        """
        logger.debug(
            'Using the {0:.2f} percentile as noise level'.format(
                self.noise_percentile
            )
        )
        noiselevel = float(
            np.nanpercentile(
                a=array,
                q=self.noise_percentile,
                interpolation="linear",
            )
        )
        return noiselevel

    def _save_noiselevel(self, noiselevel, time_obj):
        """
        Save the given noiselevel and time object pair to the attributes of this
        filter.

        Parameters
        ----------
        time_obj : numpy.datetime64
            The valid time of the given noise level.
        """
        logger.debug(
            'Saving the noise level for time {0:s}'.format(str(time_obj))
        )

        self.times = np.append(self.times, time_obj)
        self.thresholds = np.append(self.thresholds, noiselevel)

    def _prune_thresholds(self, time_obj):
        """
        Prune the thresholds based on the given time object and set remembrance
        time.

        Parameters
        ----------
        time_obj : numpy.datetime64
            This time object will be used as base time for the pruning process.
        """
        remem_date, _ = self._get_remembrance_interval(time_obj)
        pruned_thres = self.xr_thresholds.sel(times=slice(remem_date, None))
        self.thresholds = pruned_thres.values
        self.times = pruned_thres['times'].values

    def _get_last_changed_time(self):
        """
        Get the index, where the noise levels changed the last time.

        Returns
        -------
        time_obj : numpy.datetime64 or numpy.nan
            The time of the last changed value within the thresholds. If the
            index is np.nan, then no noise level was saved or the noise level
            was saved without setting the appropriate time.
        """
        try:
            different = self.thresholds != self.thresholds[-1]
            reversed_different = different[::-1]
            ind = different.size - np.argmax(reversed_different)
            ind = min(ind, different.size-1)
            time_obj = self.times[ind]
        except IndexError:
            time_obj = np.nan
        logger.debug(
            'Determined {0:s} as last changed time'.format(str(time_obj))
        )
        return time_obj

    @log_decorator(logger)
    def fit(self, array, time_obj=None):
        """
        Given a reflectivity array, determine the :any:`noiselevel` for use in
        :any:`transform`

        Parameters
        ----------
        array: array-like
            The **linear** reflectivity :math:`Z`
        time_obj : datetime.datetime or numpy.datetime64
            The time ``array`` was recorded. Defaults to ``array.time`` if
            available, otherwise :any:`datetime.datetime.now`.
        """
        time_obj = self._get_time(array, time_obj)
        first_guess = array.lawr.to_z()

        field_wo_noise = self.transform(first_guess, time_obj=time_obj)
        is_rainy = self._is_rainy_field(field_wo_noise)

        if is_rainy:
            logger.debug("Pretty much the whole dataarray suggests rain")
            noiselevel = self._get_old_noise_level(time_obj)
        else:
            noiselevel = self._determine_noiselevel_from_array(first_guess)

        logger.debug(
            'Determined noiselevel for time {0:s} is: {1:e}'.format(
                str(time_obj), noiselevel
            )
        )

        self._save_noiselevel(noiselevel, time_obj)
        self._prune_thresholds(time_obj)

        constant_for_td = time_obj - self.noiselevel_constant_since
        constant_for_s = int(constant_for_td)/1e9
        logger.debug(
            'The noise level is constant for: {0:.3f} s'.format(constant_for_s)
        )
        if constant_for_s > self.max_time_constant_noiselevel:
            self.reset()
