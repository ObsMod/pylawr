Nowcasting
==========

The :py:mod:`pylawr` package makes possible to
extrapolate weather radar fields on a very short term timescale.
The extrapolation is required if you want to process radar measurements of
several weather radars with different time stamps, e.g. our X-band weather
radars and the C-band radars operated by the German Meteorological Service
measure with different time resolutions (30 s vs. 5 min). The
:ref:`Dual radar attenuation correction` requires approximately simultaneous
reflectivity fields, therefore we need to shift the C-band measurements.

Extrapolation
-------------
The class :py:class:`~pylawr.transform.temporal.Extrapolator` can
be used to extrapolate a field based on two previous fields
using template matching (:py:func:`skimage.feature.match_template`).
The template matching finds similar areas between
two fields, for further information see
`Template Matching of scikit-image <http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html>`_.
Based on the distance between similar pixels, a vector of pixel
movement between two time steps is calculated. With the vector, the current
field can be shifted to a field of a following time step.

The :py:class:`~pylawr.transform.temporal.Extrapolator` tags
the extrapolated array with
:py:const:`~pylawr.transform.temporal.extrapolation.TAG_EXTRAPOLATION`,
which indicates that an extrapolation was applied.

.. note::

    Please note, the extrapolation is only defined for reflectivities on a
    :py:class:`~pylawr.grid.CartesianGrid`.

Some example for low-level API application is shown below. Two C-band
measurements of with different timestamps (five minutes difference)
are used to extrapolate the latest measurement to 30 seconds in the future.

.. code-block:: python

    grid_extrapolation = CartesianGrid(start=-30000, nr_points=600)

    extrapol_remapper = NearestNeighbor(1)
    extrapol_remapper.fit(dwd_grid, grid_extrapolation)

    dwd_extrapol_remapper = NearestNeighbor(1)
    dwd_extrapol_remapper.fit(grid_extrapolation, dwd_grid)

    dwd_regridded = extrapol_remapper.remap(dwd_field)
    old_dwd_regridded = extrapol_remapper.remap(old_dwd_field)

    extrapolator = Extrapolator()
    extrapolator.fit(array=dwd_regridded, array_pre=old_dwd_regridded,
                     grid=grid_extrapolation)
    extrapolator.transform(dwd_regridded,
                           time=(dwd_field.time.values[0] +
                                 np.timedelta64(30, 's')))


    dwd_extrapolated = dwd_extrapol_remapper.remap(dwd_extrapolated)

.. autosummary::
    pylawr.transform.temporal.Extrapolator

Functional API
--------------

The functional API can be used for simplified handling of the
:py:mod:`~pylawr.transform.temporal.Extrapolator` within this
:py:mod:`pylawr` package.

The online processing requires a fitted
:py:mod:`~pylawr.transform.temporal.Extrapolator` with given reflectivity
and a path to the previous reflectivity field, which is implemented within
:py:func:`~pylawr.functions.fit.fit_extrapolator`.

When processing weather radar data offline the past and future is known, it's
possible to fit an :py:mod:`~pylawr.transform.temporal.Extrapolator
for temporal interpolation between two given
arrays, which is implemented within
:py:func:`~pylawr.functions.fit.extrapolate_offline`.
The :py:mod:`~pylawr.transform.temporal.Extrapolator is applied to both fields
and then the weighted average is
returned as extrapolated field. The weights are based on a linear dynamics
assumptions and anti-proportional from array time to the interpolation time.

.. autosummary::
    pylawr.functions.fit.fit_extrapolator
    pylawr.functions.fit.extrapolate_offline

