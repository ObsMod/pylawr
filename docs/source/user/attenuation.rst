Attenuation
===========

Especially at short wavelengths, reflectivity measurements, e.g. at X- and
K-bands, are significantly attenuated by liquid water along their paths, while
attenuation at S-Band is practically negligible :cite:`lengfeld2014`.
To retrieve the intrinsic reflectivity attenuation corrections have to be
applied. For this purpose we provide two procedures, a single radar and a dual
radar attenuation correction.

Single radar attenuation correction
-----------------------------------
The single radar attenuation correction is a gate-by-gate correction developed
by :cite:`kramer2008` and :cite:`jacobi2016` based on the iterative estimation
of A(Z) relationships. For this application we can simply integrate
wradlib_ into our package with
:py:func:`wradlib.atten.correct_attenuation_constrained`, which calculates the
path integrated attenuation. :cite:`jacobi2016` evaluated this attenuation
correction for a C-band weather radar of the german weather service (DWD). The
parameters are i.a. wavelength dependent. We tuned the parameters for
C-band and for X-band by ourselves using micro rain radar (MRR) measurements
of multiple years covering 10252 fit values following
the methodology introduced by :cite:`overeem2021rainfall`.

The code below shows the simple possibility to integrate wradlib_ for single
radar attenuation correction.

.. _wradlib: https://docs.wradlib.org/en/latest/

.. code-block:: python

    from wradlib.atten import correct_attenuation_constrained, \
                              constraint_dbz, constraint_pia

    pia = correct_attenuation_constrained(refl.values,
                                          a_max=5.9e-6,
                                          a_min=2.71e-6,
                                          n_a=200,
                                          b_max=0.97,
                                          b_min=0.89,
                                          n_b=10,
                                          gate_length=0.25,
                                          constraints=[constraint_dbz,
                                                       constraint_pia],
                                          constraint_args=[
                                              [59.0], [constraint_pia_arg]]
                                          )

    pia = xr.DataArray(
        data=pia,
        coords=refl.coords,
        dims=refl.dims,
    )
    pia = pia.lawr.set_variable('pia')

    refl_corrected = refl + pia
    refl_corrected = refl_corrected.lawr.set_metadata(refl)
    tag_array(refl_corrected, "attenuation-corr-single")

.. warning::
    The function :py:func:`wradlib.atten.correct_attenuation_constrained` may
    return highly unstable results of path integrated attenuation. We added
    checks and warnings for the violation of thresholds and highly unstable
    results within the functional api
    (e.g. :py:func:`pylawr.functions.transform.correct_attenuation_lawr`).


Dual radar attenuation correction
---------------------------------
The dual radar attenuation correction
:py:class:`~pylawr.transform.attenuation.AttenuationCorrectionDual` estimates the
attenuation based on the maximum apparent attenuation comparing reflectivities
of attenuation-influenced frequency bands (X-band) and observations from less
attenuated radar systems (C-band) :cite:`lengfeld2016`.
The reflectivites have to be on the same grid.
One approach is to interpolate the reflectivites on a joint coarse grid,
estimate the attenuation and interpolate the attenuation on the fine grid of
the attenuation-influenced frequency band.

To estimate the attenuation of the attenuation-influenced frequency band
(X-band) with the less attenuated radar system (C-band) the maximum apparent
attenuation :math:`K_{\mathrm{max}}` (dB) is used:

.. math::
    K_{\mathrm{max}} &= 10\cdot{}log[z_{\mathrm{C}} / z_{\mathrm{X}}] \\
                     &= Z_{\mathrm{C}} - Z_{\mathrm{X}}

The attenuation should increase with increasing distance in theory. To get
an increasing attenuation some regression is applied on
:math:`K_{\mathrm{max}}`, e.g. the isotonic regression. For further information
look up Lengfeld et. al (2016) :cite:`lengfeld2016`. Note that the correction is
only applied where the attenuated reflectivity is available, otherwise it would
be possible that a reflectivity is created due to e.g. different radar
resoultion or incorrect alignment.

The application of the dual attenuation correction by low-level-api is shown
below with the attenuated reflectivity ``refl_attenuated`` (e.g. X-band) and
the robust reflectivity ``refl_robust`` (e.g. C-band), both are type of
:any:`xarray.DataArray`.

.. code-block:: python

    # get grids of reflectivity fields
    grid_attenuated = get_verified_grid(refl_attenuated)
    grid_robust = get_verified_grid(refl_robust)

    # define coarse grid for the calculation
    # calculation on pattern area with dwd range resolution
    nr_ranges = int(round(grid_attenuated.nr_ranges
                          * grid_attenuated.range_res
                          / grid_robust.range_res))

    grid = PolarGrid(grid_attenuated.center,
                     nr_ranges=nr_ranges,
                     range_res=grid_robust.range_res)

    # remap data on coarse grid
    remap = NearestNeighbor(1)
    remap.fit(grid_robust, grid)
    remapped_robust = remap.remap(refl_robust)
    remap.fit(grid_attenuated, grid)
    remapped_attenuated = remap.remap(refl_attenuated)

    # estimate the attenuation
    atten_corr = AttenuationCorrectionDual()
    atten_corr.fit(remapped_attenuated, remapped_robust)

    # remap attenuation data on fine grid
    remap.fit(grid, grid_attenuated)
    attenuation = remap.remap(atten_corr.attenuation)

    # correct the attenuation
    refl_corrected = refl_attenuated + attenuation
    refl_corrected = refl_corrected.lawr.set_metadata(refl_attenuated)
    tag_array(refl_corrected, "attenuation-corr-dual-isotonic")
    refl_corrected = refl_corrected.lawr.set_grid_coordinates(grid_attenuated)


.. autosummary::
    pylawr.transform.attenuation.AttenuationCorrectionDual

Functional API
--------------
For functional-api usage you have to distinguish between single and dual
attenuation correction:

* **Single radar attenuation correction**:
    We have two methods for each radar
    type - :py:func:`~pylawr.functions.transform.correct_attenuation_dwd` (C-band)
    and :py:func:`~pylawr.functions.transform.correct_attenuation_lawr` (X-band).
    Both functions only need the :any:`xarray.DataArray` of the radar reflectivity.

* **Dual radar attenuation correction**:
    The function
    :py:func:`~pylawr.functions.transform.correct_attenuation_dual` applies the
    dual attenuation correction and requires two arrays of simultaneous measured
    reflectivities of X-band and C-band. The function
    :py:func:`~pylawr.functions.transform.correct_attenuation` applies the
    single or dual radar attenuation correction depending if the second
    radar data (C-Band) is available.

.. autosummary::
    pylawr.functions.transform.correct_attenuation_lawr
    pylawr.functions.transform.correct_attenuation_dwd
    pylawr.functions.transform.correct_attenuation_dual
    pylawr.functions.transform.correct_attenuation
