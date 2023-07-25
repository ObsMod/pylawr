Grid
====
The grid capabilities are one of the main features of this :py:mod:`pylawr`
package and
are used to georeference radar fields.
This georeference is necessary for noise-correction, attenuation-correction,
remapping and plotting.
All grids have a common interface and are automatically used for different
purposes.
The most important interface is the translation of grid coordinates to latitude
and longitude coordinates.
This translation is defined within every grid type and excessively used within
:py:mod:`pylawr`.

Different grid types are defined in this submodule, which are shortly reviewed
in the following:

Polar coordinates (native radar grid)
-------------------------------------
Every point within this coordinate system is defined with an ``azimuth`` angle
and a ``range`` distance from the center point and an altitude in meters.
This grid is centered around a given ``center`` point (lat, lon, height).
Further settings of the resolution and number of grid points in azimuthal and
range direction are possible.
The altitude for every grid point is estimated based on the center height and
specified ``beam elevation``.

These polar coordinates are normally the native radar grid and almost any
non-processed radar data will be on this grid.
The pre-settings are set to the default values for the X-band radar on the
Geomatikum.

.. autosummary::
    pylawr.grid.PolarGrid


Rectangular coordinates
-----------------------
Rectangular coordinates are commonly defined by a X- and Y-position and a static
altitude.
These grids have a starting point (``start``) and ``resolution`` in its own
coordinates. The number of coordinates can be additionally defined
(``nr_points``).
The grid altitude is defined by the height of a specified
``center`` point (lat, lon, height).

Two different kinds of rectangular grid are defined at the moment:

The cartesian grid defines the X- and Y-position as relative position in meters
to a given center point (lat, lon, height).
An additional plate carre grid (:py:class:`~pylawr.grid.LatLonGrid`) defines the
X- and Y-position as absolute position in terms of latitude and longitude.

These grids are normally used to compare and combine data from different radars.
The pre-settings are set to cover the area of the X-band radar on top of the
Geomatikum.

.. autosummary::
    pylawr.grid.CartesianGrid
    pylawr.grid.LatLonGrid
    pylawr.grid.RectangularGrid


Unstructured coordinates
------------------------
Coordinates of the unstructured grid are defined by their position in latitude
and longitude, specified with an array of ``in_coords``.
An additional dimension of altitudes can be given.
If the altitudes are not given, they will be inferred from a specified
``center`` point (lat, lon, height).
In comparison to the rectangular latitude and longitude grid, this unstructured
grid follows no building structure and the coordinates have to be given
explicitly.

This grid can be used as interface to non-specified grids and is currently
only used to fill-in holes after clutter detection in operations. Further this
grid is not fully defined at the moment and can therefore only be used for
intermediate remapping purpose.

.. autosummary::
    pylawr.grid.UnstructuredGrid


Helper functions
----------------
Different helper functions are used to translate different grid definitions into
other grids.

:py:func:`~pylawr.functions.grid.get_latlon_grid` translates the latitude and
longitude coordinates of any given grid into a rectangular
:py:class:`~pylawr.grid.LatLonGrid` and infers automatically the settings of
this rectangular grid.

A similar function is :py:func:`~pylawr.functions.grid.get_cartesian`,
translating the latitude and longitude coordinates of any given grid into an
array of cartesian coordinates.
These cartesian coordinates are then relative to the center of the source grid.

An additional helper function, called
:py:func:`~pylawr.functions.grid.get_masked_grid`, masks coordinates within any
given grid.
The grid coordinates are masked based on a given mask array.
Due to masking, the resulting grid is an
:py:class:`~pylawr.grid.UnstructuredGrid`, which can be remapped to a structured
grid.

.. autosummary::
    pylawr.functions.grid.get_latlon_grid
    pylawr.functions.grid.get_cartesian
    pylawr.functions.grid.get_masked_grid


Functional API
--------------
The functional grid API can be used for a simplified handling of grids within
this :py:mod:`pylawr` package.

Only a function to remap data from a source grid to a target grid is defined at
the moment. This :py:func:`~pylawr.functions.grid.remap_data` function remaps
the data with a given remapper (see also :ref:`Remapping`). If no remapper is
given, the data is interpolated by nearest neighbor.

.. autosummary::
    pylawr.functions.grid.remap_data
