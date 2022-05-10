Load data
=========

:py:mod:`pylawr` package offers several possibilities to load data as
:py:class:`~pylawr.RadarField` (see also :ref:`Radar fields`) in post and online
processing.

Datahandler
-----------
The :py:class:`~pylawr.datahandler.base.DataHandler` is responsible for a single opened
file and has methods to decode the data, i.a. radar reflectivity and grid, from
the file. Some file types, e.g. ``.txt`` or ``.hdf5``, make implementations of
:py:class:`~pylawr.datahandler.base.DataHandler` necessary.

* The :py:class:`~pylawr.datahandler.LawrHandler` is constructed to read in LAWR text
  files, computed by online raw data processing. The file handler in the
  example below, is a filelike object, which needs to be open
  (for further information see `io — Core tools for working with streams
  <https://docs.python.org/3/library/io.html>`_).

.. code-block:: python

    data_handler = LawrHandler(file_handler)
    reflectivity = data_handler.get_reflectivity()
    reflectivity = reflectivity.lawr.set_grid_coordinates(PolarGrid())
    reflectivity.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)

* The :py:class:`~pylawr.datahandler.DWDHDF5Handler` is constructed to read in DWD HDF5
  files for single radar sites. DWD's HDF5 files are following the OPERA Data
  Information Model (ODIM) such that it should be possible to read in any ODIM
  HDF5 file. All methods are written to extract horizontal reflectivity from
  DWD radars, except
  :py:meth:`~pylawr.datahandler.DWDHDF5Handler.get_datasets`.

.. code-block:: python

    data_handler = DWDHDF5Handler(file_handler)
    read_refl = data_handler.get_reflectivity()
    grid = data_handler.grid
    read_refl = read_refl.lawr.set_grid_coordinates(grid)
    read_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)

There is no need for a netCDF file handler, see `netCDF`_.

.. autosummary::
    pylawr.datahandler.base.DataHandler
    pylawr.datahandler.DWDHDF5Handler
    pylawr.datahandler.LawrHandler

netCDF
------
As netCDF is the recommended binary serialization format for xarray object,
it's easy to read netCDF files to :py:class:`~pylawr.RadarField`, which is a
modified :any:`xarray.DataArray`. Please remind to close netCDF files after
usage.

The following example deals with loading a netCDF file written and processed by
our :py:mod:`pylawr` package, which does not need big effort.

.. code-block:: python

    dataset = xr.open_dataset(file_path, engine='netcdf4')
    reflectivity = dataset['reflectivity']
    reflectivity = reflectivity.lawr.set_grid_coordinates(PolarGrid)

The following example deals with loading a netCDF file written and processed by
another package or software, in this case our old python 2 radar package. To
load other netCDF you need to rename the units and variable names according to
our :ref:`Naming Conventions`.

.. code-block:: python

    dataset = xr.open_dataset(file_path, engine='netcdf4')

    coordinates = dict(
        time=dataset.Time.values,
        azimuth=dataset.Azimuth.values,
        range=dataset["Att_Corr_Cband_Reflectivity"].dist.values)

    attrs = dict(unit='dBZ')
    attrs.update(dataset.attrs)

    read_refl = xr.DataArray(
        data=dataset["Att_Corr_Cband_Reflectivity"].values,
        coords=coordinates,
        dims=['time', 'azimuth', 'range'],
        attrs=attrs)

    read_refl = read_refl.lawr.set_variable('reflectivity')

    read_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)

    read_refl.lawr.add_tag('old algorithms: ' + dataset.used_algorithms)

    grid = PolarGrid(center=(float(dataset.latitude[:-1]),
                             float(dataset.longitude[:-1]),
                             height),
                     beam_ele=float(dataset.elevation),
                     )

    read_refl = read_refl.lawr.set_grid_coordinates(grid)


Functional API
--------------

The :py:mod:`pylawr` package simplifies the input of all common file types in
this project. Our X-Band data is commonly distributed in ascii-data
(online) and netCDF format. Unfortunately there are more than one netCDF file
standards from our radar software, because of older software versions.
The DWD C-Band data is commonly in HDF5 format.
The functions take an opened file or a file path and a grid suitable for the
data.

.. autosummary::
    pylawr.functions.input.read_lawr_ascii
    pylawr.functions.input.read_lawr_nc_level0
    pylawr.functions.input.read_lawr_nc_old
    pylawr.functions.input.read_lawr_nc_new
    pylawr.functions.input.read_dwd_hdf5