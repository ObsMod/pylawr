#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import tempfile
import datetime
import collections
import six

# External modules
import h5py
import xarray as xr
import numpy as np

# Internal modules
from .base import DataHandler
from pylawr.grid import PolarGrid, GridNotAvailableError, avail_grids
from pylawr.utilities.decorators import lazy_property
from pylawr.utilities.conventions import naming_convention


logger = logging.getLogger(__name__)


class DWDHDF5Handler(DataHandler):
    """
    The HDF5Handler is constructed to read in DWD HDF5 files for single
    radar sites. DWD's HDF5 files are following the OPERA Data Information
    Model (ODIM) such that it should be possible to read in any ODIM HDF5
    file. All methods are written to extract horizontal reflectivity from
    DWD radars, except
    :py:meth:`pylawr.datahandler.hdf5.DWDHDF5Handler.get_datasets`.

    Parameters
    ----------
    fh : filelike object
        The data is read from this filelike object.
    """
    def __init__(self, fh):
        super().__init__(fh)
        self._valid_data = dict(
            object=[b'PVOL', b'SCAN', 'PVOL', 'SCAN'],
            version=[b'H5RAD 2.2', b'H5rad 2.2', b'H5RAD 2.1', b'H5rad 2.1',
                     'H5RAD 2.2', 'H5rad 2.2', 'H5RAD 2.1', 'H5rad 2.1']
        )
        self._attrs_keys = ['what', 'where', 'how']
        self._grid = None
        self._metadata = None
        self._file = None

    @lazy_property
    def file(self):
        self.fh.seek(0)
        read_data = self.fh.read()
        hdf5_file = self._bytes_to_hdf5(read_data)
        if not self.hdf5_validation(hdf5_file):
            raise ValueError('This hdf5 data handler is not configured to load '
                             'this data!')
        return hdf5_file

    @staticmethod
    def _bytes_to_hdf5(byte_data):
        """
        Method to read in a byte string as HDF5-file.
        This is based on: https://stackoverflow.com/a/45900556/4750376

        Returns
        -------
        hdf5_file : h5py.File
            The opened HDF5-file with data from given byte string and an
            temporary filename, which doesn't exist.

        Warnings
        --------
        This method is not back-checked with unittest!
        """
        file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        file_access_property_list.set_fapl_core(backing_store=False)
        file_access_property_list.set_file_image(byte_data)
        file_id_args = {
            'fapl': file_access_property_list,
            'flags': h5py.h5f.ACC_RDONLY,
            'name': next(tempfile._get_candidate_names()).encode(),
        }
        h5_file_args = {'backing_store': False, 'driver': 'core', 'mode': 'r'}
        file_id = h5py.h5f.open(**file_id_args)
        hdf5_file = h5py.File(file_id, **h5_file_args)
        return hdf5_file

    def hdf5_validation(self, hdf5_file):
        """
        Method to validate an opened HDF5 file. It is checked if `object` and
        `version` attributes within `what` group are the same as specified.
        Parameters
        ----------
        hdf5_file : :py:class:`h5py.File`
            An opened HDF5 file, which should be validated.

        Returns
        -------
        valid : bool
            If the given HDF5 file is valid or not.
        """
        valid_crit = [
            hdf5_file['what'].attrs[crit] in self._valid_data[crit]
            for crit in self._valid_data.keys()
        ]
        return all(valid_crit)

    @staticmethod
    def _decode_datetime_from_node(node):
        """
        Decode `startdate` and `starttime` string from given node into a
        datatime object.

        Parameters
        ----------
        node : :py:class:`h5py.Group`
            The datetime is decoded from this dataset. `startdate` and
            `starttime` of the `what` node of this dataset are used to decode
            the datetime object.

        Returns
        -------
        ds_datetime : :py:class:`numpy.datetime64`
            The decoded dataset datetime as `numpy.datetime64` object, because
            xarray uses `numpy.datetime64` to represent datetime data. The
            timezone UTC is assumed as specified in the ODIM documentation.
        """
        ds_date = node['what'].attrs['startdate']
        ds_time = node['what'].attrs['starttime']
        ds_datetime_str = '{0:s}_{1:s}'.format(
            ds_date.decode('UTF-8'), ds_time.decode('UTF-8')
        )
        ds_datetime = datetime.datetime.strptime(
            ds_datetime_str, '%Y%m%d_%H%M%S'
        )   # timezone is already UTC
        ds_datetime = np.datetime64(ds_datetime)
        return ds_datetime

    def _get_center(self):
        """
        Get the center from global where attributes within the HDF5 file.

        Returns
        -------
        lat : float
            Longitude position of the radar antenna (degrees), normalized to the
            WGS-84 reference ellipsoid and datum. Fractions of a degree are
            given in decimal notation.
        lon : float
            Latitude position of the radar antenna (degrees), normalized to the
            WGS-84 reference ellipsoid and datum. Fractions of a degree are
            given in decimal notation.
        height : float
            Height of the centre of the antenna in meters above sea level.
        """
        lat = self.file['where'].attrs['lat']
        lon = self.file['where'].attrs['lon']
        height = self.file['where'].attrs['height']
        return lat, lon, height

    def _get_grid_data(self, node):
        """
        Extract informations of the grid

        Parameters
        ----------
        node : :py:class:`h5py.Group`
            The grid data is extracted for this HDF5 group. The group needs a
            `where` node with `nrays`, `nbins`, `rscale`, `rstart`, `elangle` as
            attributes. If these attributes cannot be extracted, a
            GridNotAvailableError will be raised.

        Returns
        -------
        grid_data : dict
            The grid dict, which can be used to construct a
            :py:class:`~pylawr.grid.polar.PolarGrid`.
        """
        try:
            grid_data = dict(
                center=self._get_center(),
                nr_azi=node['where'].attrs['nrays'],
                nr_ranges=node['where'].attrs['nbins'],
                range_res=node['where'].attrs['rscale'],
                range_offset=node['where'].attrs['rstart'],
                beam_ele=node['where'].attrs['elangle']
            )
        except KeyError:
            raise GridNotAvailableError(
                'Necessary information to construct a PolarGrid cannot be '
                'extracted from the given node {0:s}'.format(str(node))
            )
        return grid_data

    def _get_metadata_from_node(self, node):
        """
        Decode the metadata from given dataset into a dictionary.

        Parameters
        ----------
        node : :py:class:`h5py.Group`
            The metadata is extracted for this HDF5 group. To extract the
            metadata the attributes of `/`, `where`, `what`, `how` are used.

        Returns
        -------
        metadata : dict(str)
            The extracted metadata from the given node. All keys with an array
            as value will be deleted. All byte values will be decoded into
            `UTF-8` strings.
        """
        raw_metadata = dict()
        metadata = dict()
        avail_groups = [
            node.get(k) for k in self._attrs_keys if k in node.keys()
        ]
        for group in avail_groups:
            group_metadata = dict(group.attrs)
            raw_metadata.update(group_metadata)
        for key, value in raw_metadata.items():
            is_iterable = isinstance(value, collections.abc.Iterable) and \
                          not isinstance(value, six.string_types)
            if isinstance(value, bytes):
                metadata[key] = value.decode('UTF-8')
            elif not is_iterable:
                metadata[key] = value
        return metadata

    def _get_dataarray_from_node(self, node):
        """
        Get a :py:class:`xarray.DataArray` from given node.

        Parameters
        ----------
        node : :py:class:`h5py.Group`
            The :py:class:`xarray.DataArray` is created for this HDF5 node. The
            data is extracted from `data` within this node.

        Returns
        -------
        dataarray : :py:class:`xarray.DataArray`
            The extracted dataarray from given node. All correction mechanisms
            with `gain`, `offset`, `nodata` and `undetect` are already applied.
        """
        metadata = self._get_metadata_from_node(node)
        node_data = np.array(node['data']).astype(float)
        node_data *= metadata.get('gain', 1.)
        node_data += metadata.get('offset', 0.)
        no_data = metadata.get('nodata', -99999)
        node_data[node_data == no_data] = np.nan
        name = metadata.get('quantity', None)
        dataarray = xr.DataArray(node_data, attrs=metadata, name=name)
        return dataarray

    def _get_grid_from_node(self, node):
        try:
            grid_data = self._get_grid_data(node)
            grid = PolarGrid(**grid_data)
        except GridNotAvailableError:
            grid = None
        return grid

    @staticmethod
    def _define_data_coords(dataarray, valid_dt=None, grid=None):
        """
        Define the coordinates of the given dataarray.

        Parameters
        ----------
        dataarray : :py:class:`xarray.DataArray`
            The coordinates of this data array will be set.
        valid_dt : :py:class:`numpy.datetime64` or None
            If this datetime is set, then a `time` axis will be prepended with
            this datetime as value.
        grid : child of :py:class:`~pylawr.grid.base.BaseGrid`
            If this grid is set, then
            :py:meth:`~pylawr.field.RadarField.set_grid_coordinates` with this
            grid will be called.

        Returns
        -------
        dataarray : :py:class:`xarray.DataArray`
            The data array with the set coordinates.
        """
        if isinstance(valid_dt, np.datetime64):
            dataarray = dataarray.expand_dims('time', axis=0)
            dataarray['time'] = [valid_dt, ]
            dataarray['time'].attrs = naming_convention['time']
        if isinstance(grid, avail_grids):
            dataarray = dataarray.lawr.set_grid_coordinates(grid)
        return dataarray

    def _get_dataset_from_node(self, node):
        ds_data = dict()
        ds_metadata = self._get_metadata_from_node(node)
        valid_dt = self._decode_datetime_from_node(node)
        avail_data = [node.get(k) for k in node.keys() if 'data' in k]
        grid = self._get_grid_from_node(node)
        for data_node in avail_data:
            data = self._get_dataarray_from_node(data_node)
            data = self._define_data_coords(data, valid_dt, grid)
            ds_data[data.name] = data
        dataset = xr.Dataset(data_vars=ds_data, attrs=ds_metadata)
        return dataset

    def get_datasets(self):
        """
        Get all datasets within the hdf5 file as :py:class:`xarray.Dataset`. The
        data nodes of the datasets are transformed into
        :py:class:`xarray.DataArray`.

        Returns
        -------
        datasets : dict(:py:class:`xarray.Dataset`)
            A dict with all extracted datasets from this HDF5 file. The data
            nodes are transformed into :py:class:`xarray.DataArray` with
            `quantity` as their array name. The attributes within where, what
            and how nodes are transformed into attributes of the datasets resp.
            dataarrays.
        """
        root_metadata = self._get_metadata_from_node(self.file)
        avail_datasets = [k for k in self.file.keys() if 'dataset' in k]
        datasets = dict()
        for node_name in avail_datasets:
            xr_ds = self._get_dataset_from_node(self.file.get(node_name))
            xr_ds.attrs.update(root_metadata)
            datasets[node_name] = xr_ds
        return datasets

    def _get_available_dates(self):
        dataset = self.file['dataset1']
        return [self._decode_datetime_from_node(dataset), ]

    def get_reflectivity(self, var='DBZH'):
        """
        Get the reflectivity from this array.

        Parameters
        ----------
        var : str
            The variable name of the reflectivity. This variable need to have
            dBZ as unit within the ODIM HDF5-file. This variable has a lower
            priority than `data_name`. `

        Returns
        -------
        reflectivity : :py:class:`xarray.DataArray`
            The decoded reflectivity from `dataset1` within the HDF5 file in
            dBZ. If possible, the grid is set for this reflectivity.
        """
        datasets = self.get_datasets()
        reflectivtiy = datasets['dataset1'][var]
        reflectivtiy = reflectivtiy.lawr.set_grid_coordinates(self.grid)
        reflectivtiy = reflectivtiy.lawr.set_variable('dbz')
        return reflectivtiy

    @lazy_property
    def grid(self):
        dataset = self.file['dataset1']
        grid_data = self._get_grid_data(dataset)
        grid = PolarGrid(**grid_data)
        return grid
