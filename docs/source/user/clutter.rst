Clutter
=======

The :py:mod:`pylawr` package comes with useful filters for adjusting radar data
according to effects of non-meteorological echoes (Clutter).
Clutter is characterised by high reflectivity values and is
caused by static objects, e.g. trees and houses, or moving objects, e.g. insects
and plains. It varies from time step to time step and from range gate to range
gate. For further general information see :cite:`lengfeld2014`.

Concept
-------

:py:class:`~pylawr.transform.filter.clutterfilter.ClutterFilter` calculates a map based on the filters
mathematic attributes. For example the :py:class:`~pylawr.transform.filter.TDBZFilter`
computes the average of the squared logarithmic reflectivity difference between
adjacent range gates. Based on the computed map and filter specific thresholds
a :py:class:`~pylawr.transform.filter.ClutterMap` is created, which classifies if the radar
signal is clutter or not in the range gate. Back to the example of our
:py:class:`~pylawr.transform.filter.TDBZFilter`, if the mean of squared reflectivity
difference within five consecutive range gates is higher than 3 dBZ, the range
gate is flagged as clutter. The derived clutter can be removed from the
reflectivity field with the method
:py:meth:`~pylawr.transform.filter.ClutterMap.transform`. of the class
:py:class:`~pylawr.transform.filter.ClutterMap`

.. code-block:: python

    tdbz_filter = TDBZFilter(window_size=5, threshold=3.)
    tdbz_clt = tdbz_filter.create_cluttermap(reflectivity)
    refl_filtered = tdbz_filter.transform(reflectivity)

Despite the application of clutter filters, some clutter remains in the radar
reflectivity. Several algorithms are applied to identify clutter signals and
interferences. To combine the results of several
:py:class:`~pylawr.transform.filter.clutterfilter.ClutterFilter`, use the
:py:class:`~pylawr.transform.filter.ClutterMap`. You can set a fuzzy threshold, which is a
relative value to the number of clutter maps, which indicate clutter, to fulfill
clutter condition. Note, if you apply more than one
:py:class:`~pylawr.transform.filter.clutterfilter.ClutterFilter` of same type, add some name
extension with the parameter `add_name`, otherwise you would overwrite one
clutter map with the append function (see below).

.. code-block:: python

    tdbz_filter = TDBZFilter(window_size=5, threshold=3.)
    tdbz_filter_two = TDBZFilter(window_size=11, threshold=30.)
    spin_filter = SPINFilter(threshold=3., window_size=11, window_criterion=.1)

    tdbz_clt = tdbz_filter.create_cluttermap(reflectivity)
    tdbz_clt_two = tdbz_filter_two.create_cluttermap(reflectivity,
                                                     addname='_some_more_info')
    spin_clt = spin_filter.create_cluttermap(reflectivity)

    cluttermap = ClutterMap('ClutterMapName', fuzzy_threshold=0.5)
    cluttermap.append(tdbz_clt)
    cluttermap.append(tdbz_clt_two)
    cluttermap.append(spin_clt)

    refl_filtered = cluttermap.transform(reflectivity)


Filter algorithms
-----------------
In the following you can find the implemented clutter filter algorithms.

TDBZ filter
^^^^^^^^^^^
This filter calculates the texture of the logarithmic reflectivity (TDBZ)
according to Hubbert et al. (2009) :cite:`hubbert2009` modified to 1D
computations. The TDBZ field is computed as the average of the squared
logarithmic reflectivity difference between adjacent range gates:

.. math::
    \mathrm{TDBZ} = \left[ \sum_{i}^{N} (\mathrm{dBZ}_{i} -
                            \mathrm{dBZ}_{i-1})^2 \right] / N

where :math:`\mathrm{dBZ}` is reflectivity and :math:`N` is the number of range
gates used. If the mean within five consecutive range gates exceeds
:math:`3\,\mathrm{dBZ}` (default parameters), the range gate is flagged as
clutter :cite:`lengfeld2014`.

.. autosummary::
    pylawr.transform.filter.TDBZFilter

SPIN filter
^^^^^^^^^^^
This filter calculates SPIN change of the reflectivity according to
Hubbert et al. (2009) :cite:`hubbert2009` modified to 1D computations. The SPIN
field is a measure of how often the reflectivity gradient changes sign along the
radial direction, with following conditions:

.. math::
    \mathrm{sign}\left(X_i - X_{i-1}\right) =
    \mathrm{sign}\left(X_{i+1} - X_i\right)

and

.. math::
    \frac{|X_i - X_{i-1}| + |X_{i+1} - X_i|}{2} > \mathrm{spinthres}

where :math:`X_{i+1}`, :math:`X_{i}`, :math:`X_{i-1}` are three consecutive dBZ
values along a radar radial. The number of sign changes is calculated within a
window of 11 range gates around the centre range gate and the reflectivity
threshold is :math:`5\,\mathrm{dBZ}` (default parameters) :cite:`lengfeld2014`.

.. autosummary::
    pylawr.transform.filter.SPINFilter

Spike filter
^^^^^^^^^^^^
The spike filter identifies clutter in the form of spikes by calculating the
reflectivity gradients for consecutive radar beams. If the differences in radar
reflectivity of adjacent radar beams to the beam of interest exceed a threshold,
the gate satisfies the first condition for clutter detection. If this condition
is fulfilled for a certain percentage of consecutive range gates within a
window, the gate is identified as clutter. Summarized, the following conditions
need to be fulfilled for a percentage of e.g. :math:`50\,\%` within a window of
e.g. eleven consecutive range gates, with a theshold of :math:`3\,\mathrm{dBZ}`
and with a spike width :math:`W` of one:

.. math::
    X_i - X_{i-W} > \mathrm{thres}

and

.. math::
    X_i - X_{i+W} > \mathrm{thres}

The index is for different radar beams.

We recommend to use two spike filters with a spike with of one and two to
identify spikes effectively. The spike and ring filters are quite similar,
but are defined for different axes.

.. autosummary::
    pylawr.transform.filter.SPKFilter

Ring filter
^^^^^^^^^^^
The ring filter identifies clutter in the form of rings by calculating the
reflectivity gradients for consecutive range gates. The ring filter is similar
to the spike filter, but is defined for range gates instead of beams.
Summarized, the conditions of the :ref:`Spike filter` are with the index for
different ranges and need to be fulfilled
for a percentage of e.g. :math:`50\,\%` within a window of e.g. eleven
consecutive radar beams with a theshold of
:math:`3\,\mathrm{dBZ}` and with a ring width :math:`W` of one.

We recommend to use two ring filters with a ring with of one and two to
identify rings effectively.

.. autosummary::
    pylawr.transform.filter.RINGFilter

Speckle filter
^^^^^^^^^^^^^^
The speckle filter assumes that rain pixels are connected and larger than a
single pixel. Following this, the probability is high that single rain pixels
are clutter. The clutter filter looks for a number of rain pixels within a
two-dimensional neighborhood of size :math:`k x l`. If this number of rain
pixels is lower than a given threshold :math:`t`, the center pixel is identified
as clutter. This can be formulated in the following way, with :math:`X_{i,j}` as
reflectivity in dBZ at position :math:`i` and :math:`j`:

.. math::
    \sum_{i=1}^{k}\sum_{j=1}^{l} \mathrm{I}(X_{i,j}>5~\text{dBZ}) < t

.. autosummary::
    pylawr.transform.filter.SpeckleFilter

Temporal filter
^^^^^^^^^^^^^^^
The temporal filter assumes that rain is moving slowly compared of the radar
image update frequency, while clutter normally "jumps" around. We therefore can
compare the current rain image to a history of rain images to identify clutter.
If the sum of rain pixels in the last :math:`n` images of a given grid point
:math:`p` is lower than a given threshold :math:`t` (normally :math:`n=t`), then
this rain pixel is identified as clutter:

.. math::
    \sum_{i=1}^{n} \mathrm{I}(X_{p}^{i}>5~\text{dBZ}) < t

.. autosummary::
    pylawr.transform.filter.TemporalFilter

Using external filters
^^^^^^^^^^^^^^^^^^^^^^
:py:mod:`pylawr` package is able to integrate external filters. For example the
clutter filter by Gabella et. al (2002) :cite:`gabella2002` is an integration
of wradlib_. We use :py:func:`wradlib.clutter.filter_gabella` to detect clutter
signals, see example below. For further information read wradlib_ documentation.

.. _wradlib: https://docs.wradlib.org/en/latest/

.. code-block:: python

    gabella = wradlib.clutter.filter_gabella(reflectivity.values[0], wsize=5,
                                             thrsnorain=0., tr1=6., n_p=8,
                                             tr2=1.3, rm_nans=False,
                                             radial=False,
                                             cartesian=False)[None, ...]
    gabella_clt = ClutterMap('GabellaFilter', gabella.astype(int))
    refl_filtered = gabella_clt.transform(reflectivity)


Functional API
--------------

For functional-api usage please note the methods
:py:func:`~pylawr.functions.transform.remove_clutter_lawr` and
:py:func:`~pylawr.functions.transform.remove_clutter_dwd`. The clutter detection
methods are tuned for the different radar types.

.. autosummary::
    pylawr.functions.transform.remove_clutter_lawr
    pylawr.functions.transform.remove_clutter_dwd