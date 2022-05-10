#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from copy import deepcopy
from unittest.mock import patch

# External modules
import numpy as np
import xarray as xr
from wradlib.clutter import filter_gabella

# Internal modules
from pylawr.field import array_has_tag
from pylawr.functions.transform import remove_noise, remove_clutter_dwd, \
    remove_clutter_lawr, interpolate_missing, correct_attenuation_dwd, \
    correct_attenuation_lawr, correct_attenuation_dual, correct_attenuation, \
    extrapolation_offline
import pylawr.functions.transform as transform_funcs
from pylawr.field import tag_array
from pylawr.grid.polar import PolarGrid
from pylawr.grid.cartesian import CartesianGrid
from pylawr.transform.temporal.noiseremover import NoiseRemover
from pylawr.transform.spatial.beamexpansion import BeamExpansion, \
    TAG_BEAM_EXPANSION_CORR
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.spatial.interpolation import Interpolator
from pylawr.transform.temporal.extrapolation import Extrapolator
from pylawr.remap.nearest import NearestNeighbor
from pylawr.remap import OrdinaryKriging
from pylawr.remap.base import BaseRemap
from pylawr.utilities.helpers import create_array
from pylawr.utilities.conventions import naming_convention


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestAttenuationDual(unittest.TestCase):
    def setUp(self):
        self.grid_fine = PolarGrid(nr_ranges=200, range_res=100)
        self.grid_coarse = PolarGrid(nr_ranges=100, range_res=200)
        self.refl_dwd = create_array(self.grid_coarse, 30)
        self.atten = np.arange(self.grid_coarse.nr_ranges) * 20 / \
                     self.grid_coarse.nr_ranges
        refl_pattern = self.refl_dwd - self.atten
        self.remapper = NearestNeighbor()
        self.remapper.fit(self.grid_coarse, self.grid_fine)
        self.refl_pattern = self.remapper.remap(refl_pattern)
        self.tag = "attenuation-corr-dual-isotonic"

    def test_corrects_to_unattenuated(self):
        correct_refl = self.remapper.remap(self.refl_dwd)
        returned_refl, _ = correct_attenuation_dual(self.refl_pattern,
                                                    self.refl_dwd)
        xr.testing.assert_equal(returned_refl, correct_refl)

    def test_sets_tag(self):
        returned_refl, _ = correct_attenuation_dual(self.refl_pattern,
                                                    self.refl_dwd)
        self.assertIn(self.tag,
                      returned_refl.attrs['tags'])

    def test_correct_attenuation_uses_dual_if_dwd(self):
        right_array, _ = correct_attenuation_dual(self.refl_pattern,
                                                  self.refl_dwd)
        returned_array, _ = correct_attenuation(self.refl_pattern,
                                                refl_dwd=self.refl_dwd)
        xr.testing.assert_identical(right_array, returned_array)

    def test_correction_keeps_grid(self):
        transformed_array, _ = correct_attenuation_dual(self.refl_pattern,
                                                        self.refl_dwd)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.refl_pattern.lawr.grid)

    def test_correction_keeps_attributes(self):
        tag_array(self.refl_pattern, 'test-tag')

        transformed_array, _ = correct_attenuation_dual(self.refl_pattern,
                                                        self.refl_dwd)

        tag_array(self.refl_pattern, self.tag)

        self.assertTrue(self.refl_pattern.attrs == transformed_array.attrs)

    def test_correction_dual_returns_pia_with_attrs(self):
        pia_name = "pia"
        _, pia = correct_attenuation_dual(self.refl_pattern,
                                          self.refl_dwd)
        print(pia.attrs)
        print(naming_convention[pia_name])
        self.assertEqual(pia.name, pia_name)
        self.assertDictEqual(pia.attrs, naming_convention[pia_name])

    def test_correction_use_dual_returns_pia(self):
        returned_array, pia = correct_attenuation(self.refl_pattern,
                                                  refl_dwd=self.refl_dwd)
        self.assertEqual(pia.name, 'pia')

    def test_correction_use_single_returns_pia(self):
        returned_array, pia = correct_attenuation(self.refl_pattern)
        self.assertEqual(pia.name, 'pia')


class TestAttenuationSingle(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()
        self.array = create_array(self.grid, const_val=10)
        self.tag = "attenuation-corr-single"

    def test_pia_highly_unstable(self):
        with self.assertLogs(logging.getLogger(), level='ERROR') as cm:
            array = create_array(self.grid, const_val=np.inf)
            returned_array, _ = correct_attenuation_lawr(array)
            self.assertFalse(array_has_tag(returned_array, self.tag))
            self.assertEqual(cm.output,
                             ['ERROR:pylawr.functions.transform:The PIA is '
                              'highly unstable and the attenuation correction '
                              'is not applied!'])

    def test_pia_is_added_to_reflectivity_lawr(self):
        returned_array, pia = correct_attenuation_lawr(self.array)
        correct_array = self.array + pia
        xr.testing.assert_equal(returned_array, correct_array)

    def test_pia_is_added_to_reflectivity_dwd(self):
        returned_array, pia = correct_attenuation_dwd(self.array)
        correct_array = self.array + pia
        xr.testing.assert_equal(returned_array, correct_array)

    def test_tag_is_added_dwd(self):
        returned_array, _ = correct_attenuation_dwd(self.array)
        self.assertIn(self.tag, returned_array.attrs['tags'])

    def test_correct_attenuation_uses_single_if_no_dwd(self):
        right_array, _ = correct_attenuation_lawr(self.array)
        returned_array, _ = correct_attenuation(self.array, refl_dwd=None)
        xr.testing.assert_identical(right_array, returned_array)

    def test_lawr_correction_keeps_grid(self):
        transformed_array, _ = correct_attenuation_lawr(self.array)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.array.lawr.grid)

    def test_lawr_correction_keeps_attributes(self):
        tag_array(self.array, 'test-tag')

        transformed_array, _ = correct_attenuation_lawr(self.array)

        tag_array(self.array, self.tag)

        self.assertTrue(self.array.attrs == transformed_array.attrs)

    def test_dwd_correction_keeps_grid(self):
        transformed_array, _ = correct_attenuation_dwd(self.array)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.array.lawr.grid)

    def test_dwd_correction_keeps_attributes(self):
        tag_array(self.array, 'test-tag')

        transformed_array, _ = correct_attenuation_dwd(self.array)

        tag_array(self.array, self.tag)

        self.assertTrue(self.array.attrs == transformed_array.attrs)

    def test_correction_lawr_returns_pia_with_attrs(self):
        pia_name = "pia"
        _, pia = correct_attenuation_lawr(self.array)
        self.assertEqual(pia.name, pia_name)
        self.assertDictEqual(pia.attrs, naming_convention[pia_name])

    def test_correction_dwd_returns_pia_with_attrs(self):
        pia_name = "pia"
        _, pia = correct_attenuation_dwd(self.array)
        print(pia.attrs)
        self.assertEqual(pia.name, pia_name)
        self.assertDictEqual(pia.attrs, naming_convention[pia_name])

class TestOfflineExtrapolation(unittest.TestCase):
    def setUp(self):
        self.grid = CartesianGrid()
        self.array = create_array(self.grid, 0)

        self.array.values[0][100:120, 100:120] = 7.
        self.array.values[0][160:180, 160:180] = 5.
        self.array.values[0][300:320, 160:180] = 10.
        self.array.values[0][160:180, 300:330] = 10.

        self.yx = [6, 6]
        self.shift = self.array.copy()
        self.shift = self.shift.roll(y=self.yx[0], x=self.yx[1],
                                     roll_coords=False)
        self.shift = self.shift.fillna(0)
        self.shift = self.shift.lawr.set_grid_coordinates(self.grid)

        self.array['time'] = [np.datetime64("2017-09-14 13:30:00")]
        self.shift['time'] = [np.datetime64("2017-09-14 13:35:00")]

        self.e = Extrapolator()

    def test_offline_extrapolated_constant_field(self):
        now = np.datetime64("2017-09-14 13:32:30")

        self.e.fit(array=self.shift, array_pre=self.array,
                   grid=self.grid)
        offline_extrapolated = extrapolation_offline(now=now,
                                                     prev_refl=self.array,
                                                     next_refl=self.shift,
                                                     extrapolator=self.e)

        transformed_array = self.e.transform(self.array, time=now)

        # the centre of the offline extrapolated array should be equal to the
        # centre of the left extrapolated array in this case
        np.testing.assert_array_equal(transformed_array.values[
                                        self.yx[0]:-self.yx[0],
                                        self.yx[1]:-self.yx[1]],
                                      offline_extrapolated.values[
                                        self.yx[0]:-self.yx[0],
                                        self.yx[1]:-self.yx[1]]
                                      )

    def test_offline_extrapolation_keeps_grid(self):
        now = np.datetime64("2017-09-14 13:32:30")

        self.e.fit(array=self.shift, array_pre=self.array,
                   grid=self.grid)
        transformed_array = extrapolation_offline(now=now,
                                                  prev_refl=self.array,
                                                  next_refl=self.shift,
                                                  extrapolator=self.e)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.array.lawr.grid)

    def test_offline_extrapolation_keeps_attributes(self):
        tag_array(self.array, 'test-tag')

        now = np.datetime64("2017-09-14 13:32:30")

        self.e.fit(array=self.shift, array_pre=self.array,
                   grid=self.grid)
        transformed_array = extrapolation_offline(now=now,
                                                  prev_refl=self.array,
                                                  next_refl=self.shift,
                                                  extrapolator=self.e)

        self.assertIn('test-tag', transformed_array.attrs["tags"])


class TestFilterFunctions(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()
        self.array = create_array(self.grid, const_val=10)
        self.array.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
        self.array = self.array.lawr.set_grid_coordinates(self.grid)

    def test_remove_noise_removes_noise(self):
        noise_filter = NoiseRemover()
        beam_expansion_filter = BeamExpansion()
        refl_final = beam_expansion_filter.transform(self.array.lawr.to_z())
        noise_filter.fit(refl_final)
        refl_final = noise_filter.transform(refl_final)
        refl_final = beam_expansion_filter.transform(refl_final.lawr.to_z())
        returned_refl, _ = remove_noise(self.array)
        xr.testing.assert_identical(returned_refl, refl_final)

    def test_remove_noise_uses_given_noise_remover(self):
        noise_filter = NoiseRemover()
        old_noise = deepcopy(noise_filter.noiselevel)
        _, returned_filter = remove_noise(
            self.array, noise_remover=noise_filter
        )
        self.assertEqual(id(returned_filter), id(noise_filter))
        self.assertNotEqual(noise_filter.noiselevel, old_noise)

    def test_remove_noise_returns_new_fitted_noise_filter_if_no_given(self):
        _, returned_filter = remove_noise(
            self.array, noise_remover=None
        )
        self.assertIsInstance(returned_filter, NoiseRemover)
        self.assertNotEqual(returned_filter.noiselevel,
                            returned_filter.default_noiselevel)

    def test_remove_noise_adds_beam_expansion_corrected(self):
        noise_filter = NoiseRemover()
        beam_expansion_filter = BeamExpansion()
        refl_final = beam_expansion_filter.transform(self.array.lawr.to_z())
        noise_filter.fit(refl_final)
        refl_final = noise_filter.transform(refl_final)
        refl_final = beam_expansion_filter.transform(refl_final.lawr.to_z())
        self.array.lawr.remove_tag(TAG_BEAM_EXPANSION_CORR)
        returned_refl, _ = remove_noise(
            self.array, noise_remover=None
        )
        xr.testing.assert_identical(returned_refl, refl_final)

    def test_remove_noise_skipsfirst_beam_if_already_applied(self):
        noise_filter = NoiseRemover()
        beam_expansion_filter = BeamExpansion()
        refl_final = beam_expansion_filter.transform(self.array.lawr.to_z())
        returned_refl, _ = remove_noise(
            refl_final, noise_remover=None
        )

        noise_filter.fit(refl_final)
        refl_final = noise_filter.transform(refl_final)
        refl_final = beam_expansion_filter.transform(refl_final.lawr.to_z())
        xr.testing.assert_identical(returned_refl, refl_final)

    def test_remove_noise_keeps_grid(self):
        transformed_array, _ = remove_noise(self.array)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.array.lawr.grid)

    def test_remove_noise_keeps_attributes(self):
        tag_array(self.array, 'test-tag')

        transformed_array, _ = remove_noise(self.array)

        self.assertIn('test-tag', transformed_array.attrs["tags"])

    def test_remove_clutter_returns_cluttermap_lawr(self):
        returned_refl, cluttermap = remove_clutter_lawr(self.array)
        self.assertIsInstance(cluttermap, ClutterMap)

    def test_remove_clutter_apply_cluttermap_lawr(self):
        returned_refl, cluttermap = remove_clutter_lawr(self.array)
        transformed_refl = cluttermap.transform(self.array)
        xr.testing.assert_identical(returned_refl, transformed_refl)

    def test_remove_clutter_returns_used_cluttermap_lawr(self):
        returned_refl, cluttermap = remove_clutter_lawr(self.array)
        self.assertIsInstance(cluttermap, ClutterMap)
        filter_names = ['TDBZFilter', 'SPINFilter']
        for n in filter_names:
            self.assertIn(n, cluttermap.layers.keys())
        right_refl = cluttermap.transform(self.array)
        xr.testing.assert_identical(returned_refl, right_refl)

    def test_remove_clutter_removes_clutter_dwd(self):
        gabella = filter_gabella(self.array.values[0], wsize=5, thrsnorain=0.,
                                 tr1=6., n_p=8, tr2=1.3, rm_nans=False,
                                 radial=False, cartesian=False)[None, ...]
        gabella_clt = ClutterMap('GabellaFilter', gabella.astype(int))
        refl_final = gabella_clt.transform(self.array)
        returned_refl, _ = remove_clutter_dwd(self.array)
        xr.testing.assert_identical(returned_refl, refl_final)

    def test_remove_clutter_returns_used_cluttermap_dwd(self):
        returned_refl, cluttermap = remove_clutter_dwd(self.array)
        self.assertIsInstance(cluttermap, ClutterMap)
        filter_names = ['GabellaFilter']
        for n in filter_names:
            self.assertIn(n, cluttermap.layers.keys())
        right_refl = cluttermap.transform(self.array)
        xr.testing.assert_identical(returned_refl, right_refl)

    def test_lawr_remove_clutter_keeps_grid(self):
        transformed_array, _ = remove_clutter_lawr(self.array)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.array.lawr.grid)

    def test_lawr_remove_clutter_keeps_attributes(self):
        tag_array(self.array, 'test-tag')

        transformed_array, _ = remove_clutter_lawr(self.array)

        self.assertIn('test-tag', transformed_array.attrs["tags"])

    def test_dwd_remove_clutter_keeps_grid(self):
        transformed_array, _ = remove_clutter_dwd(self.array)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.array.lawr.grid)

    def test_dwd_remove_clutter_keeps_attributes(self):
        tag_array(self.array, 'test-tag')

        transformed_array, _ = remove_clutter_dwd(self.array)

        self.assertIn('test-tag', transformed_array.attrs["tags"])

    def test_interpolate_missing_removes_nans(self):
        refl_nans, _ = remove_clutter_lawr(self.array)
        remapper = NearestNeighbor(5)

        int_filter = Interpolator(threshold=0.95, algorithm=remapper,)
        refl_final = refl_nans.lawr.to_dbz()
        refl_final = int_filter.transform(refl_final)

        returned_refl, _ = interpolate_missing(refl_nans, remapper=remapper)
        xr.testing.assert_allclose(refl_final, returned_refl)

    def test_interpolate_returns_used_remapper(self):
        refl_nans, _ = remove_clutter_lawr(self.array)
        _, returned_remapper = interpolate_missing(refl_nans)
        self.assertIsInstance(returned_remapper, BaseRemap)

    def test_interpolate_uses_given_remapper(self):
        remapper = NearestNeighbor(1)
        refl_nans, _ = remove_clutter_lawr(self.array)
        _, returned_remapper = interpolate_missing(refl_nans, remapper=remapper)
        self.assertIsInstance(returned_remapper, type(remapper))
        self.assertEqual(returned_remapper.n_neighbors, remapper.n_neighbors)
        self.assertEqual(returned_remapper.max_dist, returned_remapper.max_dist)

    def test_interpolate_keeps_grid(self):
        transformed_array, _ = interpolate_missing(self.array)

        self.assertTrue(transformed_array.lawr.grid ==
                        self.grid)

    def test_interpolate_keeps_attributes(self):
        tag_array(self.array, 'test-tag')

        transformed_array, _ = interpolate_missing(self.array)

        self.assertIn('test-tag', transformed_array.attrs["tags"])

    def test_interpolate_missing_deepcopies_remapper(self):
        remapper = NearestNeighbor(1)
        _, new_remapper = interpolate_missing(
            self.array, remapper
        )
        self.assertNotEqual(id(remapper), id(new_remapper))

    def test_set_max_trunc_checks_if_kernel_remapper(self):
        remapper = NearestNeighbor(1, max_dist=20)
        remapper = transform_funcs._set_max_trunc_radius(remapper, 2)
        self.assertEqual(remapper.max_dist, 20)

    def test_set_max_trunc_checks_if_trunc_is_none(self):
        remapper = OrdinaryKriging(max_dist=123456789)
        remapper = transform_funcs._set_max_trunc_radius(remapper, None)
        self.assertEqual(remapper.max_dist, 123456789)

    def test_set_max_sets_max_dist_to_trunc_decorr(self):
        remapper = OrdinaryKriging(max_dist=123456789)
        decorr_param = remapper.kernel.get_named_param('decorrelation')[0]
        decorr_param.value = 100
        remapper = transform_funcs._set_max_trunc_radius(remapper, 2)
        self.assertEqual(remapper.max_dist, 200)
        remapper = transform_funcs._set_max_trunc_radius(remapper, 1.5)
        self.assertEqual(remapper.max_dist, 150)

    def test_set_max_checks_if_decorrelation(self):
        remapper = OrdinaryKriging(max_dist=123456789)
        decorr_param = remapper.kernel.get_named_param('decorrelation')[0]
        decorr_param.value = 100
        decorr_param.name = 'length'
        remapper = transform_funcs._set_max_trunc_radius(remapper, 2)
        self.assertEqual(remapper.max_dist, 123456789)

    def test_interpolate_missing_uses_set_max_trunc(self):
        remapper = NearestNeighbor(1, max_dist=10000)
        with patch('pylawr.functions.transform._set_max_trunc_radius',
                   return_value=remapper) as set_patch:
            _ = interpolate_missing(self.array, remapper=remapper,
                                    trunc_radius=5)
        set_patch.assert_called_once()
        self.assertIsInstance(set_patch.call_args[0][0], NearestNeighbor)
        self.assertEqual(set_patch.call_args[0][0].n_neighbors, 1)
        self.assertEqual(set_patch.call_args[0][0].max_dist, 10000)
        self.assertEqual(set_patch.call_args[0][1], 5)


if __name__ == '__main__':
    unittest.main()
