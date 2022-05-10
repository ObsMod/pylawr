Noise
=====

Reflectivity measurements are disturbed by noise from internal electrical
circuits used in the receiver chain and by atmospheric noise from outside the
system :cite:`lengfeld2014`. To be able to dectect also weak weather signals an
accurate background noise estimation is crucial.

Beam Expansion
--------------
In contrast to the received signal, the signal of background noise is
independent of the distance to the radar. To estimate the noise level we need
to apply the beam expansion effect, more precisely we have to invert the
preprocessed :math:`r^2`-dependency of the background noise. For this purpose
the :py:mod:`pylawr` package provides the class
:py:class:`~pylawr.transform.spatial.BeamExpansion`. This filter applies or
removes the beam expansion effect to or from a reflectivity field. The
linear reflectivity to operate on has to be tagged with
:py:const:`~pylawr.transform.spatial.beamexpansion.TAG_BEAM_EXPANSION_CORR` or
:py:const:`~pylawr.transform.spatial.beamexpansion.TAG_BEAM_EXPANSION_UNCORR`
accordingly. Keep in mind to remove the beam expansion effect after noise
removal, otherwise your radar reflectivity decreases with :math:`r^2`.

The example below shows, some simple application of the filter
:py:class:`~pylawr.transform.spatial.BeamExpansion`.

.. code-block:: python

    if not array_has_tag(reflectivity, TAG_BEAM_EXPANSION_UNCORR):
        refl_filtered = beam_expansion_filter.transform(
            reflectivity.lawr.to_z(), inverse=True
    )


Remove Noise
------------
The :py:class:`~pylawr.transform.temporal.NoiseRemover` determines dynamically
an appropriate spatially independent noise threshold to subtract from the
reflectivity field. In a first step, an initial guess of noise level from
a rain-free field is used to separate assumed meteorological signals from noise
background. The initial first guess overestimates the expected noise level by
approximately a factor of 10, so that no noise will remain in the radar image
after noise removal. If more than 10 % radar range gates detect no rain
after subtracting the initial guess from the original reflectivity field,
the 10th percentile of the original reflectivity field is chosen as the next
noise level. If this condition is not fulfilled, the noise level from the last
time step is kept. The estimated noise level is used as the initial guess for
the noise estimation in the next time step. To reduce influence of radar
artefacts on the algorithm, the average of the recent 10 estimates is used to
correct the reflectivity. Note the beam expansion before and after noise
removal (:ref:`Beam Expansion`). For additional information please see
:cite:`lengfeld2014`.

Two tags come with the class
:py:class:`~pylawr.transform.temporal.NoiseRemover`. The tag
:py:const:`~pylawr.transform.temporal.noiseremover.TAG_NEG_LIN_REFL_REPL`
indicates that negative linear reflectivities were replaced with something.
The tag :py:const:`~pylawr.transform.temporal.noiseremover.TAG_NOISE_FILTER`
indicate that a noise filter was applied.

The application of the filter is exemplary shown below.

.. code-block:: python

    # Treat `NoiseRemover` as dynamic object, so do not instantiate this class
    # inside a for-loop processing multiple time steps.
    noise_remover = NoiseRemover()

    beam_expansion_filter = BeamExpansion()

    # The array `reflectivity` should be tagged with `TAG_BEAM_EXPANSION_CORR`
    if not array_has_tag(reflectivity, TAG_BEAM_EXPANSION_UNCORR):
        refl_filtered = beam_expansion_filter.transform(
            reflectivity.lawr.to_z(), inverse=True
        )
    else:
        refl_filtered = reflectivity.lawr.to_z()
    noise_remover.fit(refl_filtered)
    refl_filtered = noise_remover.transform(refl_filtered)
    refl_filtered = beam_expansion_filter.transform(refl_filtered.lawr.to_z())

    # Keep in mind, the filtered reflectivity is linear, so for later usage
    # you might convert it to logarithmic reflectivity.
    refl_filtered = refl_filtered.lawr.to_dbz()

.. autosummary::
    pylawr.transform.spatial.BeamExpansion
    pylawr.transform.temporal.NoiseRemover


Functional API
--------------

For functional-api usage please note the method
:py:func:`~pylawr.functions.transform.remove_noise`. The method removes
background noise by taking an instance of a
:py:class:`~pylawr.transform.temporal.NoiseRemover` and the radar field. The
advantage of this method is, you do not need to consider beam expansion, if the
tag for beam expansion exists.

.. autosummary::
    pylawr.functions.transform.remove_noise
