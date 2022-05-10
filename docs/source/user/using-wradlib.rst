Using wradlib
=============

We want a customized package, but we do not want to reinvent the wheel! It's
quite useful to integrate other packages into the :py:mod:`pylawr` package.
The most popular python package to process weather radar data is wradlib_
:cite:`wradlib`. So why are we not using just wradlib? One aspect is that
at the very beginning of the development of :py:mod:`pylawr` wradlib was not
as well-developed as today, but the main aspect is we are facing other challenges
due to our networked system architecture using X- and C-band weather radars.
Additionally the low-cost single polarised X-band weather radars require more effort
in preprocessing more background noise and clutter remains in the measurements
compared to professional weather radar systems. All in all the :py:mod:`pylawr`
package provides missing implementations, e.g. for online processing and flexible
plotting routines. Nonetheless we want use benefits from both packages. Wradlib
provides suitable algorithms for clutter detection
(see :ref:`Using external filters`) and attenuation correction
(see :ref:`Single radar attenuation correction`),
which we are using. Some wradlib application is shown below.

.. _wradlib: https://docs.wradlib.org/en/latest/

.. code-block:: python

    gabella = wradlib.clutter.filter_gabella(reflectivity.values[0], wsize=5,
                                             thrsnorain=0., tr1=6., n_p=8,
                                             tr2=1.3, rm_nans=False,
                                             radial=False,
                                             cartesian=False)[None, ...]
    gabella_clt = ClutterMap('GabellaFilter', gabella.astype(int))
    refl_filtered = gabella_clt.transform(reflectivity)

