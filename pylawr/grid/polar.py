#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
from collections import OrderedDict

# External modules
import numpy as np
import xarray as xr

# Internal modules
from pylawr.grid.base import BaseGrid
from pylawr.utilities.conventions import naming_convention


logger = logging.getLogger(__name__)


class PolarGrid(BaseGrid):
    """
    The PolarGrid is the natural grid of a radar. The polar grid is defined
    such that the radar is the north pole of the grid. Within the
    initialization, the ranges and azimuth are valid for the lower left grid
    boundary. Later, the methods are valid for the centroid.

    Parameters
    ----------
    center : tuple(float), optional
        The center of the grid. The center should have two or three entries
        within the tuple. The first entry is the latitude position of the
        grid center in degree. The second entry is the longitude position of
        the grid center in degree. The optional third entry is the height of
        the center above mean sea level in meters. If no third entry is
        given the center height is set to zero. The center is
        used to transform the polar coordinates into latitude and longitude
        coordinates. It is also used to calculate the altitude map together
        with the beam elevation. Default is the position on  the top of the
        "Geomatikum": (53.56796, 9.97451, 95).
    beam_ele : float, optional
        The beam elevation in degrees. The beam elevation is used to
        calculate the altitude map. The beam elevation is set constant if
        it is a float. The value of the beam elevation are normalized for
        angle between -90 and 90 degrees such that the sign is preserved.
        This beam elevation is used to calculate the elevation for a grid
        point. The calculated elevation is valid for the whole grid box.
    nr_ranges : int, optional
        The number of ranges with every beam. Default is 333.
    range_res : float, optional
        The step width of every range in meters. Default is 59.95849.
    range_offset : float, optional
        The offset of the ranges in meters. This value is used as start
        value for the creation of the offsets. Default is 0.
    nr_azi : int, optional
        The number of azimuth angles. This is used to calculate the azimuth
        angles. Default is 360.
    azi_offset : float, optional
        The azimuth offset in degrees. This value is used as offset for the
        creation of the azimuth linspace. This could be used to rotate the
        radar grid. Default is 0.
    """
    def __init__(self, center=(53.56796, 9.97451, 95), beam_ele=3,
                 nr_ranges=333, range_res=59.95849, range_offset=0, nr_azi=360,
                 azi_offset=0):
        super().__init__(center, beam_ele)
        self._coord_names = ('azimuth', 'range',)
        self._data_shape = (nr_azi, nr_ranges)
        self._align = None
        self._align_convert = OrderedDict(
            left=0,
            center=0.5,
            right=1,
            bottom=0,
            top=1
        )
        self.nr_ranges = nr_ranges
        self.range_res = range_res
        self.nr_azi = nr_azi
        self.range_offset = range_offset
        self.azi_offset = azi_offset

    @property
    def center_distance(self):
        return self.coord_fields[1]

    @property
    def beam_elevation_field(self):
        if isinstance(self._beam_elevation, np.ndarray):
            repeated_field = self._beam_elevation.repeat(self.nr_ranges)
            repeated_field = repeated_field.reshape(self._data_shape)
            return repeated_field
        else:
            return super().beam_elevation_field

    @property
    def ranges(self):
        return self.coords[1]

    @property
    def azimuth(self):
        return self.coords[0]

    def _calc_coordinates(self):
        azi_centroid_off = 360 / self.nr_azi / 2
        range_centroid_off = self.range_res / 2
        azimuth = np.linspace(0, 360, self.nr_azi, endpoint=False,
                              dtype=np.float64)
        azimuth += (self.azi_offset + azi_centroid_off)
        azimuth = azimuth % 360
        range_space = np.arange(self.nr_ranges, dtype=np.float64)
        range_space *= self.range_res
        range_space += (self.range_offset + range_centroid_off)
        return azimuth, range_space

    def _coords2latlon(self, azimuth, range):
        """
        Method to convert the polar coordinate field of a radar to latitude and
        longitude coordinates. The formulas are taken from
        `Wikipedia-Nautisches-Dreieck
        <https://de.wikipedia.org/wiki/Nautisches_Dreieck>`_.

        Parameters
        ----------
        range_field : Field containing the range values
        azimuth_field : Field containing the angle information in degree.

        Returns
        -------
        latitude : numpy.ndarray
            The calculated latitude coordinate as numpy.ndarray with the shape
            of the coordinates.
        longitude : numpy.ndarray
            The calculated longitude coordinate as numpy.ndarray with the shape
            of the coordinates.

        """
        lat_center = np.deg2rad(self.center[0])
        lon_center = np.deg2rad(self.center[1])

        a = np.deg2rad(-(180 + azimuth))
        h = np.deg2rad(90) - range / self.earth_radius

        delta = np.arcsin(-np.cos(h) * np.cos(a) * np.cos(lat_center) +
                          np.sin(h) * np.sin(lat_center)
                          )

        tau = np.arcsin(np.cos(h) * np.sin(a) / np.cos(delta))

        latitude = np.rad2deg(delta)
        longitude = np.rad2deg(lon_center + tau)

        return latitude, longitude

    def _coords_bounds(self):
        azimuth = np.linspace(0, 360, self.nr_azi+1, endpoint=True,
                              dtype=np.float64)
        azimuth += self.azi_offset
        azimuth = azimuth % 360
        range_space = np.arange(self.nr_ranges+1) * self.range_res
        range_space += self.range_offset
        return azimuth, range_space

    def get_azimuth_offset(self):
        """
        Get the azimuth offset for this grid as `xarray.DataArray`.

        Returns
        -------
        elevation : `xarray.DataArray`
            The azimuth offset for this grid as xarray.DataArray in degrees.
        """
        da_name = 'azimuth_offset'
        azimuth_offset = xr.DataArray(
            data=self.azi_offset,
            coords=None,
            dims=None,
            name=da_name,
            attrs=naming_convention[da_name]
        )
        return azimuth_offset
