#!/bin/env python
# -*- coding: utf-8 -*-
# System modules
import logging
import unittest
import os

# External modules
import xarray as xr
import numpy as np
from skimage.feature import match_template


# Internal modules
from pylawr.field import tag_array
from pylawr.transform.temporal.extrapolation import Extrapolator
from pylawr.grid.polar import PolarGrid
from pylawr.grid.cartesian import CartesianGrid
from pylawr.utilities.helpers import create_array


logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class ExtrapolatorTest(unittest.TestCase):
    def setUp(self):
        """
        Explanation...
        ... in this set up are two radar images on a cartesian grid with some
        rectangles as exemplary rain areas. The two radar images are five
        minutes apart.

        array = "data from last timestep"
        shift = "data of actual timestep"
        """

        self.grid = CartesianGrid()
        self.array = create_array(self.grid, 0)

        self.array.values[0][100:120, 100:120] = 7.
        self.array.values[0][160:180, 160:180] = 5.
        self.array.values[0][300:320, 160:180] = 10.
        self.array.values[0][160:180, 300:330] = 10.

        self.shift = self.array.copy()
        self.shift = self.shift.lawr.set_grid_coordinates(self.grid)

        self.array['time'] = [np.datetime64("2017-09-14 13:30:00")]
        self.shift['time'] = [np.datetime64("2017-09-14 13:35:00")]

        self.e = Extrapolator()

    def testCalcMatchMatrix(self):
        self.shift = self.shift.roll(x=20, y=20, roll_coords=False)
        self.shift = self.shift.lawr.set_grid_coordinates(self.grid)

        cp = self.e.cut_percentage

        cv = [int(round(cp * self.array.values[0].shape[0])),
              int(round(cp * self.array.values[0].shape[1]))]

        match_mat = match_template(
            np.nan_to_num(self.shift.values[0]),
            np.nan_to_num(self.array.values[0][cv[0]:-cv[0], cv[1]:-cv[1]])
        )
        np.testing.assert_array_equal(
            match_mat, self.e.calc_match_matrix(self.shift, self.array)
        )

    def testFitted(self):
        self.e.fit(self.shift, self.array)
        self.assertTrue(self.e.fitted)

    def testReset(self):
        self.e.fit(self.shift, self.array)
        self.e.reset()
        self.assertFalse(self.e.fitted)

    def testFitMovement(self):
        movement = [[5, 10], [-5, 10], [-5, -10], [10, 5]]

        for yx in movement:
            shift = self.shift.roll(y=yx[0], x=yx[1], roll_coords=False)
            shift = shift.lawr.set_grid_coordinates(self.grid)

            self.e.fit(shift, self.array)

            timedelta = (self.shift.time.values[0] -
                         self.array.time.values[0]
                         ) / np.timedelta64(1, 's')

            dist = self.e.vector * timedelta / shift.lawr.grid.resolution

            np.testing.assert_array_equal(yx, dist)

    def testFitNan(self):
        yx = [5, 10]

        self.array.values[0][120:130, 120:130] = np.nan
        self.shift.values[0][180:190, 180:190] = np.nan

        shift = self.shift.roll(y=yx[0], x=yx[1], roll_coords=False)
        shift = shift.lawr.set_grid_coordinates(self.grid)

        self.e.fit(shift, self.array)

        timedelta = (self.shift.time.values[0] - self.array.time.values[0]) / \
            np.timedelta64(1, 's')

        dist = self.e.vector * timedelta / shift.lawr.grid.resolution

        np.testing.assert_array_equal(yx, dist)

    def testTransformMovement(self):
        movement = [[5, 10], [5, -10]]

        for yx in movement:
            now = self.shift.roll(y=yx[0], x=yx[1], roll_coords=False)
            now = now.lawr.set_grid_coordinates(self.grid)

            self.e.fit(now, self.array)

            array_next = self.e.transform(
                now, time=np.datetime64("2017-09-14 13:40:00")
            )

            next_image = now.roll(y=yx[0], x=yx[1], roll_coords=False)

            np.testing.assert_array_equal(
                array_next.values[0][
                    abs(yx[0]):(array_next.shape[1] - abs(yx[0])),
                    abs(yx[1]):(array_next.shape[2] - abs(yx[1]))
                ],
                next_image.values[0][
                    abs(yx[0]):(next_image.shape[1] - abs(yx[0])),
                    abs(yx[1]):(next_image.shape[2] - abs(yx[1]))
                ]
            )

    def test_transform_keeps_grid(self):
        yx = [5, 10]
        now = self.shift.roll(y=yx[0], x=yx[1], roll_coords=False)
        now = now.lawr.set_grid_coordinates(self.grid)

        self.e.fit(now, self.array)

        array_next = self.e.transform(
            now, time=np.datetime64("2017-09-14 13:40:00")
        )

        self.assertTrue(array_next.lawr.grid ==
                        now.lawr.grid)

    def test_transform_keeps_attributes(self):
        yx = [5, 10]
        now = self.shift.roll(y=yx[0], x=yx[1], roll_coords=False)
        now = now.lawr.set_grid_coordinates(self.grid)
        tag_array(now, 'test-tag')

        self.e.fit(now, self.array)

        array_next = self.e.transform(
            now, time=np.datetime64("2017-09-14 13:40:00")
        )

        self.assertIn('test-tag', array_next.attrs["tags"])

    def testTransformScaled(self):
        yx = [20, 20]

        shift = self.shift.roll(y=yx[0], x=yx[1], roll_coords=False)
        shift = shift.lawr.set_grid_coordinates(self.grid)

        next_image = self.shift.roll(y=int(0.5*yx[0]), x=int(0.5*yx[1]),
                                     roll_coords=False)
        next_image = next_image.lawr.set_grid_coordinates(self.grid)

        self.e.fit(shift, self.array)

        array_next = self.e.transform(self.shift,
                                      time=np.datetime64(
                                          "2017-09-14 13:37:30")
                                      )

        np.testing.assert_array_equal(
            array_next.values[0][
                abs(yx[0]):(array_next.shape[1] - abs(yx[0])),
                abs(yx[1]):(array_next.shape[2] - abs(yx[1]))
            ],
            next_image.values[0][
                abs(yx[0]):(next_image.shape[1] - abs(yx[0])),
                abs(yx[1]):(next_image.shape[2] - abs(yx[1]))
            ]
        )

    def testCheckGridsPolarCartesian(self):
        grid_now = CartesianGrid()
        grid_pre = PolarGrid()

        with self.assertRaises(TypeError):
            self.e._check_grids(grid_now, grid_pre)

    def testCheckGridsPolarPolar(self):
        grid_now = PolarGrid()
        grid_pre = PolarGrid()

        with self.assertRaises(TypeError):
            self.e._check_grids(grid_now, grid_pre)

    def testCheckGridsDifferentGrids(self):
        grid_now = CartesianGrid()
        grid_pre = CartesianGrid(center=(1, 1, 1))

        with self.assertRaises(ValueError):
            self.e._check_grids(grid_now, grid_pre)

    def testDetermineTimeDelta(self):
        d = self.e._determine_timedelta(np.datetime64("2017-09-14 13:32:00"),
                                        np.datetime64("2017-09-14 13:30:00"))

        self.assertEqual(d, 120.)

    def testToXarray(self):
        yx = [5, 10]

        shift = self.shift.roll(y=yx[0], x=yx[1], roll_coords=False)
        shift = shift.lawr.set_grid_coordinates(self.grid)

        self.e.fit(shift, self.array)

        ds = self.e.to_xarray()

        timedelta = (self.shift.time.values[0] -
                     self.array.time.values[0]
                     ) / np.timedelta64(1, 's')

        dist = ds["vector"].values * timedelta / shift.lawr.grid.resolution

        np.testing.assert_array_equal(yx, dist)

    def testFromXarray(self):
        self.e.fit(self.shift, self.array)
        ds = self.e.to_xarray()

        extra = Extrapolator().from_xarray(ds)
        self.assertTrue(extra.fitted)

    def testFitLowCorrelation(self):
        self.shift[:][:, :] = np.random.random(self.grid.grid_shape)
        self.shift = self.shift.lawr.set_grid_coordinates(self.grid)

        self.e.fit(self.shift, self.array)

        np.testing.assert_array_equal(self.e.vector, [0, 0])

    def test_given_grid_is_no_cartesian_raise_type_error(self):
        with self.assertRaises(TypeError) as e:
            self.e.fit(self.array, self.array, grid=PolarGrid())
        self.assertEqual('Given grid is not a CartesianGrid',
                         str(e.exception))

    def test_given_grid_checked_for_arrays(self):
        with self.assertRaises(ValueError) as e:
            self.e.fit(self.array[:, :-1], self.array, grid=self.grid)
        with self.assertRaises(ValueError) as e:
            self.e.fit(self.array, self.array[:, :-1], grid=self.grid)


if __name__ == '__main__':
    unittest.main()
