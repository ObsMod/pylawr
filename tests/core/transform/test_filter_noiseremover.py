#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import datetime
import logging
import copy
import inspect
import os
import unittest
import warnings
from contextlib import contextmanager
from unittest.mock import patch, PropertyMock

# Internal modules
from pylawr.field import tag_array
from pylawr.transform.temporal.noiseremover import NoiseRemover
from pylawr.utilities.conventions import naming_convention
from pylawr.grid.polar import PolarGrid
import pylawr.functions.input as input_funcs

# External modules
import numpy as np
import xarray as xr
import pandas as pd

logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class BasicTest(unittest.TestCase):
    def assertArraysEqual(self, a, b):
        try:
            np.testing.assert_array_almost_equal(a, b)
        except TypeError as e:
            try:
                if str(e) == "invalid type promotion":
                    self.assertTrue(
                        np.all(np.asanyarray(a) == np.asanyarray(b))
                    )
                else:
                    self.assertEqual(a, b)
            except AssertionError:
                raise AssertionError(
                    '{0} and {1} are not the same!'.format(a, b)
                )

    @contextmanager
    def assertWarningRaised(self, category=UserWarning):
        with warnings.catch_warnings(record=True) as w:
            # record all warnings
            warnings.simplefilter("always")

            # number of warnings
            nw = len(w)
            # do the test
            yield
            # check if warning was raised
            self.assertGreater(len(w), nw, "no warning raised")
            # get all raised warning categories
            warnings_raised = [x.category for x in w[nw:len(w)+1]]
            # check if warning has right category
            self.assertTrue(
                category in warnings_raised, "no {} raised".format(category)
            )

    @contextmanager
    def assertNoWarningRaised(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") # record all warnings

            # number of warnings
            nw = len(w)
            # do the test
            yield
            # check if warning was raised
            self.assertEqual(len(w), nw, "a warning was raised")


class NoiseRemoverTest(BasicTest):
    def setUp(self):
        self.kwargs = {
            "noise_percentile": 5,
            "max_noise_percentile_tendency": 0.01,
            "max_time_constant_noiselevel": 1000,
            "remembrance_time": 200,
            "noiselevel": 130,
        }

        self.kwargs_add = {
            "thresholds": [1, 100, 23],
            "times": np.array(
                [
                    "2017-01-01 11:45:49",
                    "2017-01-01 12:01:53",
                    "2017-01-01 13:24:13"
                ], dtype="datetime64[ns]"
            ),
        }

        self.f = NoiseRemover(**self.kwargs)
        self.f.thresholds = self.kwargs_add['thresholds']
        self.f.times = self.kwargs_add['times']

        self.props = [
            n for n in inspect.getfullargspec(NoiseRemover.__init__).args
            if not n == "self"
        ]

        with open(os.path.join(DATA_PATH, 'lawr_data.txt')) as fh:
            self.array, _ = input_funcs.read_lawr_ascii(fh)

class TestNoiseRemover(NoiseRemoverTest):
    def test_xr_thresholds_combines_times_and_thresholds(self):
        xr_array = xr.DataArray(
            data=self.kwargs_add['thresholds'],
            coords=dict(
                times=self.kwargs_add['times']
            ),
            dims=['times']
        )
        xr.testing.assert_identical(xr_array, self.f.xr_thresholds)

    def test_setting_noiselevel_constant_since_to_none_uses_now(self):
        self.f.times = None
        self.f.thresholds = None
        self.assertLess(
            (
                np.datetime64(datetime.datetime.now()) -
                np.datetime64(self.f.noiselevel_constant_since)
            ).astype(float),
            1e3
        )

    def test_to_xarray(self):
        p = self.f.to_xarray()
        for n in self.props:
            self.assertArraysEqual(getattr(self.f, n), p[n])


class TestNoiseRemoverQuantitatively(NoiseRemoverTest):
    def setUp(self):
        super().setUp()
        self.reflectivity = np.random.random((5,3)) * 100

    def test_get_mean_noiselevel_returns_median_noise(self):
        noise_levels = np.array([1, 2, 3, 4, 5, 6, 7])
        out_med = self.f._get_mean_noise(noise_levels)
        self.assertEqual(4, out_med)

    def test_get_mean_noiselevel_skips_nans(self):
        noise_levels = np.array([1, 2, 3, 4, 5, 6, 7, np.nan])
        out_med = self.f._get_mean_noise(noise_levels)
        self.assertEqual(4, out_med)

    def test_get_mean_noiselevel_returns_float_or_int(self):
        noise_levels = xr.DataArray(
            data=[1, 2, 3, 4, 5, 6, 7, np.nan]
        )
        out_med = self.f._get_mean_noise(noise_levels)
        self.assertEqual(4, out_med)
        self.assertIsInstance(out_med, (int, float))

    def test_determine_noiselevel_takes_time_obj(self):
        time_obj = np.datetime64("2017-01-01 13:25:13")
        out_noise = self.f._determine_noiselevel(time_obj)
        self.assertEqual(self.kwargs_add['thresholds'][-1], out_noise)

    def test_determine_noiselevel_uses_remem_time(self):
        time_obj = np.datetime64("2017-01-01 13:25:13")
        self.f.remembrance_time = 84600

        right_noise = self.f._get_mean_noise(self.kwargs_add['thresholds'])
        out_noise = self.f._determine_noiselevel(time_obj)
        self.assertEqual(right_noise, out_noise)

    def test_get_noiselevel_without_time_returns_current_noiselevel(self):
        self.assertEqual(self.f.get_noiselevel(), self.f.noiselevel)

    def test_get_noiselevel_without_set_times_returns_current_noiselevel(self):
        self.f.times = None
        self.assertEqual(self.f.times.size, 0)
        self.assertEqual(self.f.get_noiselevel(), self.f.noiselevel)

    def test_get_noiselevel_with_time_returns_the_same_as_det_noise(self):
        time_obj = np.datetime64("2017-01-01 13:25:13")
        self.f.remembrance_time = 84600
        right_noise = self.f._get_mean_noise(self.kwargs_add['thresholds'])
        out_noise = self.f.get_noiselevel(time_obj)
        self.assertEqual(right_noise, out_noise)

    def test_transform_converts_array_to_z(self):
        convention_dict = naming_convention['z']

        self.array = self.array.lawr.set_variable('dbrr')
        out_array = self.f.transform(
            self.array, time_obj=np.datetime64("2017-01-01 13:25:13")
        )
        out_attrs = {a: out_array.attrs[a] for a in out_array.attrs.keys()
                     if a in convention_dict}

        self.assertEqual(out_array.name, 'z')
        self.assertDictEqual(convention_dict, out_attrs)

    @patch.object(NoiseRemover, 'get_noiselevel', return_value=5)
    def test_transform_uses_time_from_array(self, filter_mock):
        time_obj = np.datetime64("2017-01-01 13:25:13")
        self.array['time'] = [time_obj]
        out_array = self.f.transform(self.array)

        filter_mock.assert_called_with(time_obj=time_obj)

        right_array = self.f.transform(
            self.array, time_obj=np.datetime64("2017-01-01 13:26:13")
        )
        xr.testing.assert_identical(right_array, out_array)

    @patch.object(NoiseRemover, 'get_noiselevel', return_value=5)
    def test_transform_deletes_noise(self, filter_mock):
        time_obj = np.datetime64("2017-01-01 13:25:13")
        self.array['time'] = [time_obj]
        out_array = self.f.transform(self.array, replace_negative=None)
        right_array = self.array.lawr.to_z()
        right_array -= 1.03 * 5
        xr.testing.assert_equal(right_array, out_array)

    def test_transform_deletes_negative(self):
        time_obj = np.datetime64("2017-01-01 13:25:13")
        self.array['time'] = [time_obj]
        right_array = self.f.transform(self.array, replace_negative=None)
        right_array = right_array.where(right_array > 0, np.nan)
        out_array = self.f.transform(self.array, replace_negative=np.nan)
        xr.testing.assert_equal(right_array, out_array)

    def test_transform_adds_tags(self):
        out_array = self.f.transform(self.array, replace_negative=np.nan)
        self.assertEqual(
            out_array.attrs['tags'],
            'beam-expansion-corr;neg-lin-refl-repl-nan;noise-filtered'
        )

    def test_transform_keeps_grid(self):
        self.array = self.array.lawr.set_grid_coordinates(PolarGrid())
        transformed_array = self.f.transform(self.array)
        self.assertTrue(transformed_array.lawr.grid ==
                        self.array.lawr.grid)

    def test_transform_keeps_attributes(self):
        tag_array(self.array, 'test-tag')
        transformed_array = self.f.transform(self.array)
        self.assertIn('test-tag', transformed_array.attrs["tags"])

    def test_fitted_returns_true_if_times(self):
        self.f.times = np.datetime64("2017-01-01 13:25:13")
        self.assertTrue(self.f.fitted)
        self.f.times = [
            np.datetime64("2017-01-01 13:25:13"),
            np.datetime64("2017-02-01 13:25:13")
        ]
        self.assertTrue(self.f.fitted)

    def test_fitted_returns_false_if_no_times_set(self):
        self.f._times = np.array([], dtype="datetime64[ns]")
        self.assertFalse(self.f.fitted)
        self.f.times = None
        self.assertFalse(self.f.fitted)

    def test_get_old_noise_level_returns_default_if_not_fitted(self):
        time_obj = self.f.times[-1:]
        self.f.times = None
        out_noise = self.f._get_old_noise_level(time_obj)
        self.assertEqual(out_noise, self.f.noiselevel)

    def test_get_old_noise_level_returns_latest_noise_before_time(self):
        time_obj = self.f.times[-1:]
        out_noise = self.f._get_old_noise_level(time_obj)
        self.assertEqual(out_noise, self.kwargs_add['thresholds'][-2])

    def test_determine_new_noise_takes_only_array(self):
        _ = self.f._determine_noiselevel_from_array(self.array)
        with self.assertRaises(TypeError):
            self.f._determine_noiselevel_from_array(None)

    def test_determine_new_noise_takes_linear_percentile(self):
        out_noise = self.f._determine_noiselevel_from_array(self.array)
        right_noise = np.percentile(
            self.array, self.f.noise_percentile, interpolation='linear'
        )
        self.assertEqual(out_noise, right_noise)

    def test_save_noiselevel_sets_noiselevel_to_new_value(self):
        new_noise = 2E5
        now_obj = self.f._get_time(self.array, datetime.datetime.now())
        self.f._save_noiselevel(new_noise, now_obj)
        self.assertEqual(self.f.noiselevel, new_noise)

    def test_save_noiselevel_appends_time_obj_to_times(self):
        self.assertEqual(self.f.times.size, 3)
        new_noise = 2E5
        now_obj = self.f._get_time(self.array, datetime.datetime.now())
        self.f._save_noiselevel(new_noise, now_obj)
        self.assertEqual(self.f.times.size, 4)
        self.assertEqual(self.f.times[-1], now_obj)

    def test_save_noiselevel_appends_noise_to_thresholds(self):
        self.assertEqual(self.f.thresholds.size, 3)
        new_noise = 2E5
        now_obj = self.f._get_time(self.array, datetime.datetime.now())
        self.f._save_noiselevel(new_noise, now_obj)
        self.assertEqual(self.f.thresholds.size, 4)
        self.assertEqual(self.f.thresholds[-1], new_noise)

    def test_save_noiselevel_converts_datetime_to_datetime64(self):
        new_noise = 2E5
        now_obj = datetime.datetime.now()
        self.f._save_noiselevel(new_noise, now_obj)
        self.assertIsInstance(self.f.times[-1], np.datetime64)

    def test_remembrance_date_returns_remem_date(self):
        time_obj = datetime.datetime.now()
        remembrance_delta = datetime.timedelta(
            seconds=int(self.f.remembrance_time)
        )
        remem_date = time_obj - remembrance_delta
        out_date, _ = self.f._get_remembrance_interval(time_obj)
        self.assertEqual(remem_date, out_date)

    def test_remembrance_interval_returns_time_obj_as_pd_datetime(self):
        time_obj = datetime.datetime.now()
        start_date, end_date = self.f._get_remembrance_interval(time_obj)
        self.assertIsInstance(start_date, pd.datetime)
        self.assertIsInstance(end_date, pd.datetime)
        self.assertEqual(pd.to_datetime(time_obj), end_date)

    def test_prune_thresholds_prunes_thresholds(self):
        new_noise = 2E5
        now_obj = self.f._get_time(self.array, datetime.datetime.now())
        self.f._save_noiselevel(new_noise, now_obj)
        logger.error(self.f.times)
        self.assertEqual(self.f.thresholds.size, 4)
        self.f._prune_thresholds(now_obj)
        self.assertEqual(self.f.thresholds.size, 1)
        self.assertEqual(self.f.thresholds[-1], new_noise)

    def test_prune_thresholds_prunes_times(self):
        new_noise = 2E5
        now_obj = self.f._get_time(self.array, datetime.datetime.now())
        self.f._save_noiselevel(new_noise, now_obj)
        self.assertEqual(self.f.times.size, 4)
        self.f._prune_thresholds(now_obj)
        self.assertEqual(self.f.times.size, 1)
        self.assertEqual(self.f.times[-1], now_obj)

    def test_prune_thresholds_keeps_values_from_future(self):
        time_obj = datetime.datetime(1970, 1, 1)
        old_thres = copy.deepcopy(self.f.thresholds)
        old_times = copy.deepcopy(self.f.times)

        self.f._prune_thresholds(time_obj)
        np.testing.assert_equal(old_thres, self.f.thresholds)
        np.testing.assert_equal(old_times, self.f.times)

    def test_get_last_changed_index_returns_index_of_last_change(self):
        last_changed = self.f._get_last_changed_time()
        self.assertEqual(last_changed, self.f.times[-1])

    def test_get_last_changed_index_returns_index_with_spaces(self):
        time_obj = np.datetime64(datetime.datetime.now())
        self.f.thresholds = np.array([2, 100, 2])
        right_changed = self.f.times[-1]
        self.f._save_noiselevel(2, time_obj)
        last_changed = self.f._get_last_changed_time()
        self.assertEqual(last_changed, right_changed)

    def test_get_last_changed_index_returns_nan_if_no_thresholds(self):
        self.f.thresholds = None
        last_changed = self.f._get_last_changed_time()
        np.testing.assert_equal(np.nan, last_changed)

    def test_get_last_changed_index_for_two_items(self):
        self.f.thresholds = self.f.thresholds[:2]
        self.f.times = self.f.times[:2]
        right_changed = self.f.times[-1]
        last_changed = self.f._get_last_changed_time()
        self.assertEqual(right_changed, last_changed)

    def test_get_last_changed_index_returns_single_if_single(self):
        self.f.thresholds = self.f.thresholds[:1]
        self.f.times = self.f.times[:1]
        right_changed = self.f.times[-1]
        last_changed = self.f._get_last_changed_time()
        self.assertEqual(right_changed, last_changed)

    def test_reset_resets_noiselevel_to_default(self):
        self.assertNotEqual(self.f.noiselevel, self.f.default_noiselevel)
        self.f.reset()
        self.assertEqual(self.f.noiselevel, self.f.default_noiselevel)

    def test_reset_resets_thresholds_to_empty(self):
        empty = np.array([])
        self.assertGreater(self.f.thresholds.size, 0)
        self.f.reset()
        self.assertEqual(self.f.thresholds.size, 0)
        np.testing.assert_equal(empty, self.f.thresholds)

    def test_reset_resets_times_to_empty(self):
        empty = np.array([], 'datetime64')
        self.assertGreater(self.f.times.size, 0)
        self.f.reset()
        self.assertEqual(self.f.times.size, 0)
        np.testing.assert_equal(empty, self.f.times)

    def test_is_rainy_field_uses_noise_percentile_as_thres(self):
        self.array = self.array.lawr.set_variable('dbz')
        self.array[:] = 5
        self.array[:, :180] = -5
        self.f.noise_percentile = 1
        self.assertFalse(self.f._is_rainy_field(self.array))
        self.f.noise_percentile = 80
        self.assertTrue(self.f._is_rainy_field(self.array))

    def test_is_rainy_field_transforms_to_dbz(self):
        self.array = self.array.lawr.set_variable('dbz')
        self.array[:] = -5
        self.assertFalse(self.f._is_rainy_field(self.array))
        self.array = self.array.lawr.db_to_linear()
        self.assertFalse(self.f._is_rainy_field(self.array))

    def test_is_rainy_sets_more_than_0_to_rain(self):
        self.array = self.array.lawr.set_variable('dbz')
        self.array[:] = -5
        self.assertFalse(self.f._is_rainy_field(self.array))
        self.array[:] = 0.0001
        self.assertTrue(self.f._is_rainy_field(self.array))

    @patch.object(NoiseRemover, '_get_time',
                  return_value=np.datetime64(datetime.datetime.now()))
    def test_fit_uses_get_time(self, patched):
        self.array = self.array.lawr.set_variable('dbz')
        self.f.fit(self.array)
        patched.assert_called_with(self.array, None)

    def test_fit_transforms_to_z(self):
        self.array = self.array.lawr.set_variable('dbz')
        self.f.reset()
        self.f.fit(self.array)
        old_threshold = self.f.thresholds[-1]
        self.array = self.array.lawr.to_z()
        self.f.reset()
        self.f.fit(self.array)
        new_threshold = self.f.thresholds[-1]
        self.assertEqual(old_threshold, new_threshold)

    @patch.object(NoiseRemover, 'transform',)
    def test_fit_calls_transform_for_rainy(self, patched_transform):
        patched_transform.return_value = self.array
        self.f.fit(self.array)
        patched_transform.assert_called()

    @patch.object(NoiseRemover, 'transform',)
    @patch.object(NoiseRemover, '_is_rainy_field', return_value=False)
    def test_fit_calls_is_rainy(self, patched_rainy, patched_transform):
        patched_transform.return_value = self.array
        self.f.fit(self.array)
        patched_rainy.assert_called_with(self.array)

    @patch.object(NoiseRemover, '_is_rainy_field', return_value=True)
    @patch.object(NoiseRemover, 'transform',)
    @patch.object(NoiseRemover, '_get_old_noise_level', return_value=2e-3)
    def test_fit_calls_get_old_noise_if_rainy(
        self, patched_noise, patched_transform, *args
    ):
        patched_transform.return_value = self.array
        self.f.fit(self.array)
        patched_noise.assert_called_with(self.array.time.values[0])
        self.assertEqual(self.f.thresholds[-1], 2e-3)

    @patch.object(NoiseRemover, '_is_rainy_field', return_value=False)
    @patch.object(NoiseRemover, 'transform',)
    @patch.object(NoiseRemover, '_determine_noiselevel_from_array',
                  return_value=2e-3)
    def test_fit_calls_determine_noise_if_not_rainy(
        self, patched_noise, patched_transform, *args
    ):
        patched_transform.return_value = self.array
        self.f.fit(self.array)
        patched_noise.assert_called()
        self.assertEqual(self.f.thresholds[-1], 2e-3)

    @patch.object(NoiseRemover, '_determine_noiselevel_from_array',
                  return_value=2e-3)
    def test_fit_calls_save_noiselevel(self, patched_noise):
        self.f.fit(self.array)
        self.assertEqual(self.f.thresholds[-1], 2e-3)
        self.assertEqual(self.f.times[-1], self.array.time.values[0])
        self.assertEqual(self.f.noiselevel, 2e-3)

    def test_fit_prunes_thresholds(self,):
        time_obj = np.datetime64(datetime.datetime(2019, 1, 1))

        self.assertEqual(self.f.thresholds.size, 3)
        self.f.fit(self.array, time_obj)
        self.assertEqual(self.f.thresholds.size, 1)
        self.assertEqual(self.f.times[0], time_obj)

    @patch.object(NoiseRemover, 'noiselevel_constant_since',
                  new_callable=PropertyMock,
                  return_value=np.datetime64(datetime.datetime.now()))
    def test_fit_call_last_changed_time(self, patched_time):
        self.f.fit(self.array)
        patched_time.assert_called_once()

    @patch.object(NoiseRemover, 'noiselevel_constant_since',
                  new_callable=PropertyMock,
                  return_value=np.datetime64(datetime.datetime(1970, 1, 1)))
    @patch.object(NoiseRemover, 'reset')
    def test_fit_resets_params_if_not_changed(self, reset_patch, *args):
        self.f.fit(self.array)
        reset_patch.assert_called_once()

    def test_fit_sets_right_noiselevel(self):
        self.array = self.array.lawr.to_z()
        right_noise = self.f._determine_noiselevel_from_array(self.array)
        self.f.fit(self.array)
        self.assertEqual(right_noise, self.f.noiselevel)
        self.assertEqual(right_noise, self.f.thresholds[-1])


if __name__ == '__main__':
    unittest.main()
