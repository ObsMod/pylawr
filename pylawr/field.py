#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import warnings
from functools import partial
from copy import deepcopy

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.utilities.conventions import naming_convention
from pylawr.utilities.trafo import from_decibel_to_linear, zr_vars

logger = logging.getLogger(__name__)

TAGS_KEY = "tags"
TAGS_SEP = ";"


def tag_array(array, tag):
    """
    Attempt to :py:meth:`~pylawr.RadarField.add_tag()` to a given array

    Parameters
    ----------

    array : array-like
        the array to tag
    tag : str
        the tag to add to the array

    Returns
    -------

    bool
        If tagging was successful or not
    """
    try:
        try:
            array.add_tag(tag)
        except AttributeError:
            array.lawr.add_tag(tag)
        logger.debug("Tagged array with '{}'".format(tag))
        return True
    except AttributeError:
        logger.warning("Could not tag array with '{}'!".format(tag))
        return False


def array_has_tag(array, tag):
    """
    Check if an array has a given tag

    Parameters
    ----------

    array : array-like
        the array to check
    tag : str
        the tag to look for

    Returns
    -------

    bool
        Whether the array has the tag
    """
    tags = []
    try:
        try:
            tags = array.tags
        except AttributeError:
            tags = array.lawr.tags
    except AttributeError:
        logger.warning("This array does not seem to be able to have tags!")
    return tag in tags


def untag_array(array, tag):
    """
    Remove a tag from an array with :py:meth:`~pylawr.RadarField.remove_tag()`

    Parameters
    ----------

    array : array-like
        the array to operate on
    tag : str
        the tag to remove

    Returns
    -------

    bool
        Whether the tag has been removed
    """
    try:
        try:
            array.remove_tag(tag)
        except AttributeError:
            array.lawr.remove_tag(tag)
        logger.debug("Removed tag '{}' from array".format(tag))
        return True
    except AttributeError:
        logger.warning("array does not seem to be able to handle tags!")
        return False


def get_verified_grid(array, grid=None):
    """
    Get a verified grid from the array and given grid.

    Parameters
    ----------
    array : :py:class:`xarray.DataArray`
        If no grid is given, this array will be used to determine the grid.
        The grid is verified to this array.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None
        This grid is verified within this method. If no grid is given, the
        grid from the array is used.

    Returns
    -------
    verified_grid : child of :py:class:`~pylawr.grid.base.BaseGrid`
        The verified and inferred grid. If a grid is given and verified,
        this will be the same grid as given.

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
    if grid is None:
        try:
            grid = array.lawr.grid
        except (TypeError, AttributeError):
            raise AttributeError(
                'A grid is needed, but no Grid is specified and no grid is '
                'set for the array!'
            )
    verified_grid = array.lawr.check_grid(grid=grid)
    return verified_grid


@xr.register_dataarray_accessor('lawr')
class RadarField(object):
    """
    RadarField is a xarray.DataArray accessor to extend xarray for radar
    processing. The accessor is called with xarray.DataArray.lawr.

    Parameters
    ----------
    data : :py:class:`xarray.DataArray`
        This accessor is valid for this given data array.

    Attributes
    ----------
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid`
        The grid of this RadarField. The grid is used for localization,
        interpolation and plotting purpose.
    """
    def __init__(self, data):
        self._data = data
        self._grid = None

    def __repr__(self):
        return "{}:\n{}".format(self.__class__.__name__, repr(self.data))

    @property
    def data(self):
        """
        Get the data for this accessor.

        Returns
        -------
        :py:class:`xarray.DataArray`
            The data for this :py:class:`xarray.DataArray` accessor.
        """
        return self._data

    @property
    def grid(self):
        if self._grid is None:
            raise TypeError('The grid isn\'t set yet!')
        return self._grid

    @grid.setter
    def grid(self, grid):
        logger.debug('Got {0} as grid'.format(grid))
        if grid is None:
            self._grid = grid
        else:
            self._grid = self.check_grid(grid)

    @property
    def tags(self):
        """
        Tags describing what has already happened to this field

        :type: list
        :getter: Converts the :any:`RadarField.data`
            :any:`xarray.DataArray.attrs` ``"tags"`` to a :any:`list`
        :setter: Sets the :any:`RadarField.data`
            :any:`xarray.DataArray.attrs` ``"tags"`` key
        """
        try:
            self.data.attrs[TAGS_KEY]
        except KeyError:
            self.data.attrs[TAGS_KEY] = ""
        tags = [x for x in self.data.attrs[TAGS_KEY].split(TAGS_SEP) if x != ""]
        return tags

    @tags.setter
    def tags(self, newtags):
        newtags = [x for x in newtags if x != ""]
        self.data.attrs[TAGS_KEY] = TAGS_SEP.join(newtags)

    def add_tag(self, newtag):
        """
        Add a tag to :any:`tags`

        Parameters
        ----------

        newtag : str
            The new tag to add
        """
        tags = self.tags
        if not newtag in tags:
            tags.append(str(newtag))
        self.tags = tags

    def remove_tag(self, tag):
        """
        Remove a tag (and any duplicates) from :any:`tags`. Don't complain if
        the tag doesn't exist.

        Parameters
        ----------

        tag : str
            The tag to remove
        """
        self.tags = [x for x in self.tags if x != tag]

    def set_variable(self, name):
        """
        Set the variable name of this DataArray to the given name. The unit,
        long name and short name are set according to the naming convention
        defined in `utilities/conventions.py`.

        Parameters
        ----------
        name : str
            The name of the variable. This name needs to be defined in the
            naming convention.

        Returns
        -------
        named_array : :py:class:`xarray.DataArray`
            The DataArray with the set name, unit, long name and short name. All
            other attributes will be also set. If a grid is defined for this
            DataArray it will be carried over.
        """
        if name not in naming_convention.keys():
            raise KeyError(
                'The given variable name: {0:s} cannot be found within the '
                'naming convention.\nAvailable variables: {1:s}'.format(
                    name, ','.join(naming_convention.keys())
                )
            )
        named_array = self.data.copy()
        named_array.name = name
        named_array.attrs.update(naming_convention[name])
        try:
            named_array = named_array.lawr.set_grid_coordinates(
                self.grid
            )
        except TypeError:
            # No grid is set
            pass
        return named_array

    def set_metadata(self, other_array):
        """
        Set the metadata for this array based on the given other array. The
        metadata is the name, the attributes and the grid.

        Parameters
        ----------
        other_array : :py:class:`xarray.DataArray`
            The other_array is used to extract the metadata for the data of this
            class.

        Returns
        -------
        attributed_array : :py:class:`xarray.DataArray`
            This array with replaced name, attributes and grid from other_array.
        """
        attributed_array = deepcopy(self.data)
        attributed_array.name = other_array.name
        attributed_array.attrs = other_array.attrs
        try:
            attributed_array = attributed_array.lawr.set_grid_coordinates(
                other_array.lawr.grid
            )
        except TypeError:
            # No grid is set
            pass

        return attributed_array

    def check_grid(self, grid):
        """
        Check if the given grid is a valid grid for this DataArray.

        Parameters
        ----------
        grid : child of :py:class:`~pylawr.grid.base.BaseGrid`
            This grid is checked.

        Returns
        -------
        checked_grid : child of :py:class:`~pylawr.grid.base.BaseGrid`
            The grid object is returned if it is valid.
        """
        if not hasattr(grid, '_calc_altitude'):
            raise TypeError('The grid has to be a gridlike type!')
        else:
            data_coords_shape = self._data.shape[-len(grid.grid_shape):]
            if grid.grid_shape != data_coords_shape:
                raise ValueError(
                    'This grid is has not the right coordinates shape!\nactual:'
                    '{0}, desired: {1}'.format(grid.grid_shape,
                                               data_coords_shape
                                               )
                )
            else:
                return grid

    def set_grid_coordinates(self, grid):
        """
        Set the grid coordinates for this DataArray. The coordinates of this
        DataArray are replaced by the coordinates of the grid.

        Parameters
        ----------
        grid : child of :py:class:`~pylawr.grid.base.BaseGrid`
            This grid is used to set the grid and the grid coordinates of the
            returned array.

        Returns
        -------
        gridded_array : :py:class:`xarray.DataArray`
            The :py:class:`~xarray.DataArray` with the grid coordinates and
            the grid.
        """
        _ = self.check_grid(grid)
        rename_dict = {new: old for new, old in zip(
            self.data.dims[-len(grid.coord_names):], grid.coord_names)
            if new != old}
        gridded_array = self.data.rename(rename_dict)
        new_coordinates = grid.get_coordinates()
        for coord in grid.coord_names:
            gridded_array.coords[coord] = new_coordinates[coord]
        gridded_array.lawr._grid = grid
        return gridded_array

    def grid_to_array(self):
        """
        Convert the array to a :py:class:`~xarray.DataArray` with the grid as
        attributes and variables.

        Returns
        -------
        combined_ds : :py:class:`xarray.DataArray`
            The :py:class:`~xarray.DataArray` with the added grid.
        """
        gridded_ds = self.set_grid_coordinates(self.grid).to_dataset()
        lat_lon_ds = self.grid.get_lat_lon()
        altitude_ds = self.grid.get_altitude().to_dataset(name='altitude')
        combined_ds = xr.merge([gridded_ds, lat_lon_ds, altitude_ds],)
        combined_ds = combined_ds.assign_attrs(
            grid_center=self.grid.center,
            grid_type=self.grid.__class__.__name__
        )
        return combined_ds

    def filter(self, filter_class, *args, **kwargs):
        """
        Method to apply a filter to this class.

        Parameters
        ----------
        filter_class : Python obj
            This instance of a python class is used to filter the data. The
            class needs a transform method with data as first argument and
            grid as keyword argument. The return value of the transform method
            should be a xarray.DataArray. A possible second return value is
            interpreted as new grid.
        *args :
            The variable length argument list is passed to the transform method
            of the filter_class.
        **kwargs :
            The variable keyword argument dictionary is passed to the transform
            method of the filter_class. This argument could be used if another
            array is necessary to transform this array with the given filter
            class.

        Returns
        -------
        filtered_data : xarray.DataArray
            The filtered data array, with the set grid.
        """
        filtered_data = filter_class.transform(self.data, grid=self._grid,
                                               *args, **kwargs)
        if isinstance(filtered_data, tuple) and len(filtered_data) == 2:
            filtered_data = filtered_data[0].lawr.set_grid_coordinates(
                filtered_data[1])
        elif self._grid is not None:
            filtered_data = filtered_data.lawr.set_grid_coordinates(self._grid)
        return filtered_data

    def _convert_field(self, conv_func, target_var='dbz'):
        # Catch warnings from trafo fn and transform values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            converted_field = conv_func(self.data)

        converted_field = converted_field.lawr.set_metadata(self.data)
        converted_field = converted_field.lawr.set_variable(target_var)
        return converted_field

    def zr_convert(self, a=200.0, b=1.6, inverse=False):
        r"""
        Convert from z to r via Z/R relationship. The Z/R relationship is
        calculated with

        .. math::

            r = \sqrt[b]{\frac{z}{a}}

        for conversion from z to r. If inverse is chosen, then the unit will be
        converted from r to z with:

        .. math::

            z = a \cdot r^{b}

        Parameters
        ----------
        a : float, optional
            Parameter a of the Z/R relationship. This is the scale factor of the
            Z/R relationship. Default is 200.0 from Marshall-Palmer.
        b : float, optional
            Parameter b of the Z/R relationship. This is the polynomial grade
            factor of the Z/R relationship. Default is 1.6 from Marshall-Palmer.
        inverse : bool, optional
            If the reflectivity should be calculated based on a rain rate field
            (True) or if the rain rate should be calculated based on the
            reflectivity (False). Default is False.

        Returns
        -------
        converted_field : :py:class:`xarray.DataArray`
            The converted rain field with r or z as unit and the same grid as
            this DataArray.

        Warnings
        --------
        UserWarning :
            An UserWarning will be raised if the rain field is already the right
            variable or if the field is in decibel units.
        """
        warn_msg = None

        # Get direction and target variable
        if inverse:
            conversion_func = partial(self._reflectivity, a=a, b=b)
            var = 'z'
        else:
            conversion_func = partial(self._rainrate, a=a, b=b)
            var = 'rr'

        # Check if already var
        if self.data.name == var:
            warn_msg = 'The variable is already {0:s}, I will return the ' \
                       'input array!'.format(var)

        # Check if convertible
        not_convertible = self.data.name is not None and \
            self.data.name not in zr_vars
        if not_convertible:
            warn_msg = '{0:s} cannot converted to {1:s}, I will return the ' \
                       'input array!'.format(self.data.name, var)

        # If warning
        if isinstance(warn_msg, str):
            warnings.warn(warn_msg, UserWarning)
            return self.data

        # Convert the field
        converted_field = self._convert_field(conversion_func, var)
        return converted_field

    def _rainrate(self, x, a=200.0, b=1.6):
        r"""
        Converts to rain rate r via Z/R relationship.

        .. math::
            r = \sqrt[b]{\frac{z}{a}}

        Parameters
        ----------
        x : int, float or array (radar reflectivity factor, mm**6/m**3)
        a : float, optional
            Parameter a of the Z/R relationship. This is the scale factor of the
            Z/R relationship. Default is 200.0 from Marshall-Palmer.
        b : float, optional
            Parameter b of the Z/R relationship. This is the polynomial grade
            factor of the Z/R relationship. Default is 1.6 from Marshall-Palmer.

        Returns
        -------
        converted `x` in rain rate (mm/h)
        """
        return (x / a) ** (1. / b)

    def _reflectivity(self, x, a=200.0, b=1.6):
        """
        Converts to radar reflectivity factor via Z/R relationship.

        Parameters
        ----------
        x : int, float or array (rain rate, mm/h)
        a : float, optional
            Parameter a of the Z/R relationship. This is the scale factor of the
            Z/R relationship. Default is 200.0 from Marshall-Palmer.
        b : float, optional
            Parameter b of the Z/R relationship. This is the polynomial grade
            factor of the Z/R relationship. Default is 1.6 from Marshall-Palmer.

        Returns
        -------
        converted `x` in radar reflectivity factor (mm**6/mm**3)
        """
        return a * x ** b

    def db_to_linear(self, inverse=False):
        r"""
        Transform this rain field from decibel units to linear units.

        .. math::
            \mathrm{linear} &= 10^{\frac{\mathrm{decibel}}{10}}

            \mathrm{decibel} &= 10 \cdot \log_{10}(\mathrm{linear})

        .. note::
            Due to conventions: Linear reflectivities smaller than or equal to
            :math:`0\,\mathrm{mm^6\,m^{-3}}` will be transformed to
            :math:`-32.5\,\mathrm{dBZ}` in decibel unit and vice versa.

        Parameters
        ----------
        inverse : bool, optional
            Default is False.

        Returns
        -------
        converted_field : :py:class:`xarray.DataArray`
            The converted rain field with Gaussian decibel or linear non-decibel
            units. The field has the same grid as this DataArray. The target
            variable is determined automatically. If name is None, the target
            variable will be either linear_reflectivity or reflectivity.
        """
        warn_msg = None

        # Get direction and target variable
        if inverse:
            conversion_func = self._decibel
            from_linear_to_decibel = {
                v: k for k, v in from_decibel_to_linear.items()
            }
            try:
                var = from_linear_to_decibel[self.data.name]
            except KeyError as e:
                if isinstance(self.data.name, str):
                    warn_msg = 'The variable `{0:s}` cannot be converted, I ' \
                               'will return the input array!'.format(
                                self.data.name)
                elif self.data.name is None:
                    var = 'dbz'
                else:
                    raise e
        else:
            conversion_func = self._linear
            try:
                var = from_decibel_to_linear[self.data.name]
            except KeyError as e:
                if isinstance(self.data.name, str):
                    warn_msg = 'The variable `{0:s}` cannot be converted, I ' \
                               'will return the input array!'.format(
                                self.data.name)
                elif self.data.name is None:
                    var = 'z'
                else:
                    raise e

        # Warning is raised
        if isinstance(warn_msg, str):
            warnings.warn(warn_msg, UserWarning)
            return self.data

        converted_field = self._convert_field(conversion_func, target_var=var)
        return converted_field

    def _decibel(self, x, dbzmin=-32.5):
        r"""
        Converts to decibel unit.

        .. math::

            \mathrm{decibel} = 10 \cdot \log_{10}(\mathrm{linear})

        .. note::
            Due to conventions: Linear reflectivities smaller than or equal to
            :math:`0\,\mathrm{mm^6\,m^{-3}}` will be transformed to
            :math:`-32.5\,\mathrm{dBZ}` in decibel unit.

        Parameters
        ----------
        x : :py:class:`xarray.DataArray`
            The radar field with linear non-decibel unit.

        Returns
        -------
        :py:class:`xarray.DataArray`
        The radar field with decibel unit.
        """
        mask = x <= 10. ** (dbzmin / 10.)
        return xr.where(mask, dbzmin, 10. * np.log10(x))

    def _linear(self, x, dbzmin=-32.5):
        r"""
        Converts to linear unit.

        .. math::

            \mathrm{linear} = 10^{\frac{\mathrm{decibel}}{10}}

        .. note::
            Due to conventions: Reflectivities in decibel unit smaller than or
            equal to :math:`-32\,\mathrm{dBZ}` will be transformed to
            :math:`0\,\mathrm{mm^6\,m^{-3}}` in linear unit.

        Parameters
        ----------
        x : :py:class:`xarray.DataArray`
            The radar field with decibel unit.

        Returns
        -------
        :py:class:`xarray.DataArray`
        The radar field with linear non-decibel unit.
        """
        return xr.where(x <= dbzmin, 0., 10. ** (x / 10.))

    def _z_to_z(self):
        return self._convert_field(lambda x: x, 'z')

    def _dbz_to_z(self):
        return self.db_to_linear()

    def _r_to_z(self):
        return self.zr_convert(inverse=True)

    def _dbr_to_z(self):
        return self.db_to_linear().lawr.zr_convert(inverse=True)

    def to_z(self):
        """
        Convert the rain field to linear reflectivity from any arbitrary
        variable. If no variable is specified in the attributes, it is assumed
        that the rain field is the reflectivity in decibel. If the variable of
        this DataArray is rain_rate `zr_convert` with default values will be
        used.

        Returns
        -------
        converted_field : :py:class:`xarray.DataArray`
            The converted rain field as linear reflectivity. The field has the
            same grid and attributes (except for the variable unit and names) as
            this DataArray.
        """
        from_dict = {
            None: self._dbz_to_z,
            'z': self._z_to_z,
            'dbz': self._dbz_to_z,
            'rr': self._r_to_z,
            'dbrr': self._dbr_to_z
        }
        converted_field = from_dict[self.data.name]()
        return converted_field

    def to_dbz(self):
        """
         Convert the rain field to reflectivity in decibel from any arbitrary
         variable. If no variable is specified in the attributes, it is assumed
         that the rain field is the reflectivity in decibel. If the variable of
         this DataArray is rain_rate `zr_convert` with default values will be
         used.

         Returns
         -------
         converted_field : :py:class:`xarray.DataArray`
             The converted rain field as reflectivity in dB. The field has the
             same grid and attributes (except for the variable unit and
             names) as this DataArray.
         """
        converted_field = self.to_z()
        converted_field = converted_field.lawr.db_to_linear(inverse=True)
        return converted_field

    def to_netcdf(self, save_path):
        """
        Save the RadarField to given save path as NetCDF-file. The grid is added
        to the attributes such that it can be restored from the attributes.

        Parameters
        ----------
        save_path : str
            Path where the file should be saved. If the path is a directory the
            file name is inferred by the radar and grid.
        """
        pass

    def get_rain_mask(self, threshold=5):
        mask = (self.to_dbz() > threshold).values
        return mask

    @classmethod
    def load(cls, load_path):
        """
        The RadarField is loaded from a NetCDF-file. This method should be only
        used if the field is saved with the 'to_netcdf' method.

        Parameters
        ----------
        load_path : str
            Path where the file is saved. The file need to be a NetCDF.file.

        Returns
        -------
        loaded_field : :py:class:`xarray.DataArray`
            The loaded radar field.k
        """
        pass