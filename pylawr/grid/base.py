#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import abc

# External modules
import numpy as np
import pandas as pd
import xarray as xr

# Internal modules
from pylawr.utilities.decorators import lazy_property
from pylawr.utilities.conventions import naming_convention


logger = logging.getLogger(__name__)


class BaseGrid(object):
    """
    The BaseGrid is a base class for all radar grids.

    Parameters
    ----------
    center : tuple(float), optional
        The center of the grid. The centre should have two or three entries
        within the tuple. The first entry is the latitude position of the
        grid center in degree. The second entry is the longitude position of
        the grid centre in degree. The optional third entry is the height of
        the center above mean sea level in meters. If no third entry is
        given the center height is set to zero. The centre is
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
    """
    def __init__(self, center=(53.56796, 9.97451, 95), beam_ele=3):
        self._center = None
        self._beam_elevation = None
        self._data_shape = None
        self._coord_names = None
        self._altitude = None
        self.center = center
        self.beam_elevation = beam_ele
        self._earth_radius = None

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @property
    def beam_elevation(self):
        return self._beam_elevation

    @beam_elevation.setter
    def beam_elevation(self, elevation):
        if not isinstance(elevation, (float, int)):
            raise TypeError('The given beam elevation is not a float or '
                            'an integer!')
        abs_angle = np.arcsin(
            np.abs(np.sin(elevation / 180 * np.pi))) / np.pi * 180
        self._beam_elevation = np.sign(elevation) * abs_angle

    @property
    def center(self):
        """
        The center position for this grid as tuple
        (latitude, longitude, height).
        """
        return self._center

    @center.setter
    def center(self, pos):
        if not isinstance(pos, tuple):
            raise TypeError('The given centre position is not a tuple!')
        elif len(pos) == 2:
            self._center = (*pos, 0)
        elif len(pos) == 3:
            self._center = pos
        else:
            raise ValueError('The tuple should have either 2 or 3 entries!')

    @property
    @abc.abstractmethod
    def center_distance(self):
        """
        Get the distance to the center in meters.

        Returns
        -------
        distance : numpy.ndarray
            The distance to the center in meters.
        """
        pass

    @lazy_property
    def coords(self):
        """
        The coordinates for this grid as numpy.ndarray within a tuple.
        """
        return self._calc_coordinates()

    @abc.abstractmethod
    def _calc_coordinates(self):
        """
        Calculate the coordinates for this grid.

        Returns
        -------
        coordinates : tuple(numpy.ndarray)
            The calculated coordinates for this grid.
        """
        pass

    @lazy_property
    def earth_radius(self):
        """
        The earth radius at this latitude.

        Returns
        -------
        earth_radius : float
            The earth radius in meters.
        """
        radius_equator = 6378.137   # km
        radius_pole = 6356.752  # km
        lat_rad = np.deg2rad(self.center[0])
        earth_radius = np.sqrt(
            (np.power(radius_equator, 4) * np.power(np.cos(lat_rad), 2) +
             np.power(radius_pole, 4) * np.power(np.sin(lat_rad), 2)) /
            (np.power(radius_equator * np.cos(lat_rad), 2) +
             np.power(radius_pole * np.sin(lat_rad), 2))
        )
        earth_radius = earth_radius * 1e3   # to meters
        return earth_radius

    @lazy_property
    def lat_lon(self):
        """
        The latitude and longitude coordinates for this grid.
        """
        return self._coords2latlon(*self.coord_fields)

    @abc.abstractmethod
    def _coords2latlon(self, *args):
        """
        Calculate the latitude and longitude coordinates for this grid based on
        given coordinates.

        Returns
        -------
        latitude : numpy.ndarray
            The calculated latitude coordinate as numpy.ndarray with the shape
            of the coordinates.
        longitude : numpy.ndarray
            The calculated longitude coordinate as numpy.ndarray with the shape
            of the coordinates.
        """
        pass

    @lazy_property
    def lat_lon_bounds(self):
        coord_bounds = self._coords_bounds()
        coord_bound_fields = self._calc_coord_fields(*coord_bounds)
        return self._coords2latlon(*coord_bound_fields)

    @abc.abstractmethod
    def _coords_bounds(self):
        pass

    @property
    def altitude(self):
        """
        The altitude map for this grid. Similar characteristics to a lazy
        property but the altitude is only calculated based on the beam
        elevation if the altitude was not set. Therefore, a user customized
        altitude map is possible.
        """
        if self._altitude is None:
            self.altitude = self._calc_altitude(self.center_distance,
                                                self.beam_elevation,
                                                self.center[2],
                                                self.earth_radius)
        return self._altitude

    @altitude.setter
    def altitude(self, alt):
        """
        Setter for the altitude map, which sets a altitude map according to
        the grid shape. A constant float is reshaped to a constant array.
        """
        if isinstance(alt, np.ndarray):
            if not alt.shape == self.grid_shape:
                raise ValueError(
                    'The altitude is given as numpy.ndarray, but has the '
                    'wrong shape! desired: {0}, effective: {1}'.format(
                        self.grid_shape, alt.shape)
                )
        elif isinstance(alt, (float, int)):
            alt = np.full(self.grid_shape, alt)
        else:
            raise TypeError('The given altitude is not a float or '
                            'a numpy.ndarray!')
        self._altitude = alt

    @staticmethod
    def _calc_altitude(radar_distance, beam_elevation=3.,
                       radar_height=0., earth_radius=6364335,
                       ke=4. / 3.):
        r"""
        Calculate the altitude map for this grid based on the
        distance to the radar site, the centre position height and the beam
        elevation. Calculates the radar beam height taking refraction into
        account based on :cite:`doviak2006` using

        .. math::

            h = [r^2 + (k_{\mathrm{e}}\,a)^2 + 2\,r\,k_{\mathrm{e}}\,a\,
            \sin\,\Theta_{\mathrm{e}}]^{1/2} - k_{\mathrm{e}}\,a

        with :math:`h` is the radar beam height, :math:`r` is the distance
        to the radar, :math:`k_{\mathrm{e}}` is an adjustment factor,
        :math:`a` is the earth radius, and :math:`\Theta_{\mathrm{e}}` is
        the radar elevation angle.

        Parameters
        ----------
        radar_distance : numpy.ndarray, int or float
            Distance to the radar site in metres.
        beam_elevation : float
            The beam elevation in degrees.
        radar_height : float
            Height of the radar at center position in metres.
        earth_radius : int
            Radius of earth in metres.
        ke: float
            Adjustment factor dependent on the refractive index gradient.

        Returns
        -------
        altitude : numpy.ndarray
            The calculated altitude.
        """
        altitude = (
                np.sqrt(
                    radar_distance * radar_distance
                    + ke * ke * earth_radius * earth_radius +
                    2. * radar_distance * ke * earth_radius *
                    np.sin(np.deg2rad(beam_elevation))
                ) - (ke * earth_radius) + radar_height
        )

        return altitude

    @property
    def coord_names(self):
        """
        Get the coordinate names as tuple.
        """
        return self._coord_names

    @staticmethod
    def _calc_coord_fields(*coords):
        coords = np.meshgrid(*coords)
        return tuple([c.T for c in coords])

    @property
    def coord_fields(self):
        """
        Get the coordinates as meshgrid fields.
        """
        coords = self.coords
        coord_fields = self._calc_coord_fields(*coords)
        return coord_fields

    @property
    def beam_elevation_field(self):
        """
        Get the beam elevation as repeated numpy.ndarray with the same shape as
        the data.
        """
        beam_ele_array = np.array(self._beam_elevation)
        data_size = np.product(self._data_shape)
        repeated_field = beam_ele_array.repeat(data_size)
        return repeated_field.reshape(self._data_shape)

    @property
    def grid_shape(self):
        """
        Return the shape of the grid.

        Returns
        -------
        tuple(int)
            The shape of the grid as tuple.
        """
        return self._data_shape

    @property
    def size(self):
        return np.prod(self._data_shape)

    def _check_data(self, data):
        """
        Method to check if the type and shape of the given data is right.

        Parameters
        ----------
        data : python obj
            Data to check.

        Raises
        ------
        TypeError
            TypeError is raised if the given data is not a numpy.ndarray.
        ValueError
            ValueError is raised if the given data has not the right shape.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('The given data is not a valid numpy array!')
        if np.any(np.array(data.shape) != np.array(self._data_shape)):
            raise ValueError(
                'The given data has the wrong shape! Data shape: {0}, desired '
                'shape: {1}'.format(data.shape, self._data_shape)
            )

    def get_coordinates(self):
        """
        Get the coordinates for this grid in a xarray-conform structure.

        Returns
        -------
        coords : dict(str, (str, numpy.ndarray))
            The coordinates for this grid in a xarray-conform structure. The key
            is the coordinate name. The coordinates have as value a tuple with
            their own name, indicating that the they are self-describing, and
            the coordinate values as numpy array.
        """
        coords_data = self.coords
        coords = {v: ((v, ), np.array(coords_data[k]), naming_convention[v])
                  for k, v in enumerate(self.coord_names)}
        return coords

    def get_multiindex(self):
        """
        Get the coordinates for this grid as :py:class:``pandas.Multiindex``.

        Returns
        -------
        multiindex : :py:class:``pandas.Multiindex``
            The coordinates as Multiindex with the coordinate names as level
            name. The values are the cross-product of the coordinate values.
        """
        multiindex = pd.MultiIndex.from_product(
            self.coords, names=self.coord_names
        )
        return multiindex

    def get_altitude(self):
        """
        Get the altitude map for this grid as xarray.DataArray.

        Returns
        -------
        altitude : xarray.DataArray
            The altitude map for this grid as xarray.DataArray in meters above
            mean sea level. The coordinates of the DataArray are the grid
            coordinates.
        """
        da_name = 'zsl'
        altitude = xr.DataArray(
            data=self.altitude,
            coords=self.get_coordinates(),
            dims=self.coord_names,
            name=da_name,
            attrs=naming_convention[da_name]
        )
        return altitude

    def get_lat_lon(self):
        """
        Get the latitude and longitude for this grid as xarray.Dataset.

        Returns
        -------
        lat_lon : xarray.Dataset
            The latitude and longitude coordinates as xarray.Dataset. The
            latitudes and longitudes are variables within this dataset. The
            coordinates of the dataset are the grid coordinates.
        """
        lat_lon = xr.Dataset(
            data_vars={
                'lat': (self.coord_names,
                        self.lat_lon[0],
                        naming_convention['lat']),
                'lon': (self.coord_names,
                        self.lat_lon[1],
                        naming_convention['lon']),
            },
            coords=self.get_coordinates()
        )
        return lat_lon

    def get_beam_elevation(self):
        """
        Get the beam elevation for this grid as `xarray.DataArray`.

        Returns
        -------
        elevation : `xarray.DataArray`
            The elevation for this grid as xarray.DataArray in degrees.
        """
        da_name = 'ele'
        elevation = xr.DataArray(
            data=self.beam_elevation,
            coords=None,
            dims=None,
            name=da_name,
            attrs=naming_convention[da_name]
        )
        return elevation

    def get_center(self):
        """
        Get the center lat, lon, height for this grid as `xarray.Dataset`.

        Returns
        -------
        center : `xarray.Dataset`
            The center for this grid as xarray.Dataset including
            latitude in degrees_north, longitude in degrees_east,
            and altitude in meters.
        """
        lat_center = xr.DataArray(
            data=self.center[0],
            coords=None,
            dims=None,
            name='lat_center',
            attrs=naming_convention['lat_center']
        )
        lon_center = xr.DataArray(
            data=self.center[1],
            coords=None,
            dims=None,
            name='lon_center',
            attrs=naming_convention['lon_center']
        )
        altitude_center = xr.DataArray(
            data=self.center[2],
            coords=None,
            dims=None,
            name='zsl_center',
            attrs=naming_convention['zsl_center']
        )
        center = xr.merge([lat_center, lon_center, altitude_center])
        center.attrs = {}
        return center
