Radar fields
============

:py:mod:`pylawr` ships a handy data structure, the :py:class:`~pylawr.RadarField`.
This is basically a better :py:class:`xarray.DataArray`.

Accessor
--------

As soon as you import :py:mod:`pylawr`, every :py:class:`xarray.DataArray` has
an accessor attribute ``lawr``:

.. code-block:: python

    import pylawr
    import xarray

    da = xarray.DataArray(3)
    da.lawr

This accessor returns the exact same array but as a :py:class:`~pylawr.RadarField`.

Tag system
----------

To be able to track what has happened to a :py:class:`~pylawr.RadarField`, there is a **tag
system**. Use the property :py:attr:`pylawr.RadarField.tags` to get or set the field's
tags at once. Internally they are just stored in the
:py:attr:`xarray.DataArray.attrs` dict. To add/remove single tags, use
:py:meth:`~pylawr.RadarField.add_tag` and :py:meth:`~pylawr.RadarField.remove_tag`.

There are also convenience functions :py:func:`~pylawr.field.tag_array`, :py:func:`~pylawr.field.untag_array` and
:py:func:`~pylawr.field.array_has_tag` for easy handling of multiple array types.

Array conversion
----------------

The accessor also has some additional methods to convert the :py:class:`~pylawr.RadarField`
into another variable.

It is possible to convert the reflectivity to rain rate
and vice versa (with *inverse=True*) with the :py:meth:`~pylawr.RadarField.zr_convert()` method. If no
arguments are given, then the normal Z/R relationship is assumed. To convert from
`decibel` units into linear units and vice versa, the
:py:meth:`~pylawr.RadarField.db_to_linear`
method can be used. With the :py:meth:`~pylawr.RadarField.to_z()` method, the radar field can be converted
into linear reflectivity for any set radar variable (e.g. decibel rain rate).
The :py:meth:`~pylawr.RadarField.to_dbz()` method allows to convert the radar field into reflectivity
with `dBZ` as unit.

The variable of an array can be set with the
:py:meth:`~pylawr.RadarField.set_variable()` method. Based
on the ``pylawr.utilities.conventions.naming_convention``, the metadata
(e.g. unit or short name) of the
array are set accordingly.

API References
--------------
.. autosummary::
   pylawr.RadarField
   pylawr.field.get_verified_grid
   pylawr.field.untag_array
   pylawr.field.array_has_tag
   pylawr.field.tag_array
