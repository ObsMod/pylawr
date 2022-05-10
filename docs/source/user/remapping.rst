Remapping
=========
This remapping subpackage is designed to remap (in the following interpolation
is used as synonym) data between different grids (see also :ref:`grid`).
These remappers have all a common interface and are only different in their
remapping algorithms. Remapping is implemented via two-step procedure:

1. `fit` – The remapper is fitted to remap from a source grid to a target grid.
2. `transform` – The fitted remapper transforms given radar data to the target grid.

Only tree-based remapping methods are currently implemented.
In tree-based methods, the `n`-nearest neighbors for every target grid point are
searched in the source grid.
This is done in local cartesian space spanned by the source coordinates.

Nearest-neighbor remapping
--------------------------
The simplest tree-based method is nearest neighbors interpolation.
In this nearest neighbor interpolation the median of the `n`-nearest neighbors
is used as interpolated value.
This interpolation is commonly used for interpolations from or to polar grids to
other grids.

A one-to-one interpolation is also possible with this method.
For this, the number of nearest neighbors has to be restricted to one and the
maximum interpolation radius has to be constrained to the resolution of the
source grid.

.. autosummary::
    pylawr.remap.NearestNeighbor

Kriging remapping
-----------------
A more sophisticated remapping method compared to nearest neighbor remapping is
the kriging :cite:`cressie1992`.
Kriging interpolates data by using correlations within the data and is a best
linear unbiased estimator (BLUE).
Based on these correlations it is also possible to specify the interpolation
uncertainty of the interpolated points.

It is assumed in kriging that the correlations can be modelled as
distance-depending.
For this, we implement an additional submodule allowing a wide range of different
kernels, which transforms a given distance matrix into a covariance matrix with
a set of pre-defined non-linear functions (see also :ref:`Kriging kernels`).
An interpolated point is then an weighted average of the source points with the
weights depending on given covariance matrix.

In normal kriging, one would use all available points, but for radar meteorology
this is infeasible because we have around :math:`10^5` source points.
To circumvent this problem, we use a localized version of kriging, where only
the `n`-nearest neighbors are weighted and averaged :cite:`wesson2004`. This is
also visualized in :ref:`Kriging as gaussian processes`.

It is assumed in kriging that the source values follow a random Gaussian noise
without a trend and which can be modelled by correlated fields. Two different
types of kriging are currently implemented.

The most simple case is simple kriging.
Here, we assume that the expectation is known, stationary and already subtracted
from the data.
In our case, we estimate the expectation as local mean of the `n`-nearest
neighbors.
This type of kriging is also called Gaussian Process and is used for machine
learning :cite:`rasmussen2006`.

In ordinary kriging, we only assume that the expectation is stationary and it
has to be estimate by kriging.
Further, we constrain the weights of the nearest neighbors to unity.
This form of kriging is therefore the most used form for geostatistical
interpolation.

.. autosummary::
    pylawr.remap.SimpleKriging
    pylawr.remap.OrdinaryKriging


Kriging kernels
^^^^^^^^^^^^^^^
To transform the distance matrix into a covariance matrix, we have to define a
covariance function, which is an equivalent to a variogram.
We introduce additional kernels to allow a variety of different non-linear
covariance functions.
The kernels are based on a graph paradigm.

1. The operations are planned with a composition of kernels.
2. The kernel composition is evaluated based on fed-in values. In this
    second step, the kernels are callable like a python function.


Three different type of base kernels are implemented at the moment and will be
shortly reviewed in the following.

The placeholder (:py:class:`~pylawr.remap.kernel.Placeholder`) can be used to
dynamically feed-in data into the kernel. The value of the placeholder can be
either controlled by using keyword arguments for the evaluation or the first
argument is automatically used.

The parameter (:py:class:`~pylawr.remap.kernel.Parameter`) can be used to
introduce variable and constant values into a defined kernel function. A
parameter has for every evaluation the same value, except if it is changed
externally. A parameter can be set to constant, which is then not changed during
optimization.

The white noise kernel (:py:class:`~pylawr.remap.kernel.WhiteNoise`) specifies
the observational uncertainty. The noise level of this kernel can be specified
and represents the observational variance. This white noise kernel has an
additional dependency to another kernel (e.g. placeholder) to determine the
shape of the covariance matrix for the white noise.

These three types of kernels can be manipulated by any
`numpy.ufunc <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_ such
that almost any non-linear kernel can be represented by this type of operation.
Additional to the base kernels, we defined three different types of already
composed kernels, which are commonly used for kriging or gaussian processes.

The Gaussian radial basis function kernel
(:py:class:`~pylawr.remap.kernel.gaussian_rbf`) is an universal kernel
:cite:`micchelli2006` and one of the most used kernel for Gaussian processes.
In its behaviour it can emulate almost any available function and is also called
Gaussian variogram in geostatistical literature.
The length scale of the RBF kernel specifies the decorrelation length (also
called range), while the standard deviation corresponds to the height of the
function (also called sill). The RBF kernel is so popular because of his
simplicity and flexibility. If no data-sparse regions the RBF kernel converges
to the expectation.

The exponential sine squared kernel
(:py:class:`~pylawr.remap.kernel.exp_sin_squared`) is often used to model
periodic processes. This kernel is controlled by the period and the length scale
with the same meaning as for the RBF kernel. Caused by its periodicity, this
kernel converges to wave-like solutions in data-sparse regions.

The rational quadratic kernel
(:py:class:`~pylawr.remap.kernel.rational_quadratic`) behaves like added RBF
kernels with different length scales. An additional :math:`\alpha` parameter
controls the shape of the kernel and the weighting between large-scale and
small-scale variations. This kernel converges to expectation in data-sparse
regions as the RBF kernel.

.. autosummary::
    pylawr.remap.kernel.Placeholder
    pylawr.remap.kernel.Parameter
    pylawr.remap.kernel.WhiteNoise

    pylawr.remap.kernel.gaussian_rbf
    pylawr.remap.kernel.exp_sin_squared
    pylawr.remap.kernel.rational_quadratic


Functional API
--------------
The functional api can be used to remap data from one grid to another grid.
This remapping function (:py:func:`~pylawr.functions.grid.remap_data`) fits a
remapper and then uses this remapper to remap given data.
If no remapper is given, nearest neighbor interpolation with a single neighbor
is used for remapping.

Clutter detection causes holes within the radar data.
Two different functions are pre-defined to interpolate these missing values (for
lawr: :py:func:`~pylawr.functions.transform.interpolate_missing_lawr`, for dwd:
:py:func:`~pylawr.functions.transform.interpolate_missing_dwd`).
The missing values are interpolated based on a given remapper.
The remapper is fitted to interpolate missing values and then applied to the
radar data.
If no remapper is given, ordinary kriging with a Gaussian RBF kernel and white
noise is used as default.

Kriging is an effective method for interpolation, because it can be adapted to
correlations, which prevail in the current radar field.
This adaption is normally infeasible for radar purpose because it has to solve
an optimization problem, which is time-consuming.
Here, we provide an additional function to fit a kriging method to the current
radar data based on particle filters and stochastic variogram matching (see also
:ref:`inference`).

.. autosummary::
    pylawr.functions.grid.remap_data
    pylawr.functions.transform.interpolate_missing
    pylawr.functions.fit.fit_kriging
