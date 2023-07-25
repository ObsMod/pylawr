Inference
=========
This submodule of :py:mod:`pylawr` is designed to infer parameters and rain
data from radar fields.
Inference is a process to gather an unobserved, latent state from
current and past data.
One example of inference for meteorology is data assimilation, where model
forecasts and observation data are combined to estimate the current state of
the atmosphere.
In general terms, inference can be used to estimate unknown parameters based on
observations and a known observation and forecast model.
In the following, we will explain the mathematical foundation for inference,
the implementations and how inference can be used for radar meteorology.

As only inference implementation at the moment, this package is suitable to
infer parameters for kriging kernels based on the SIR particle filter and
variogram matching.
For kriging interpolation, we need to define a kernel composition.
This kernel composition represents local distance-depending correlations and
could have several tuning parameters.
These tuning parameters can be time-depending and have to be inferred from
current radar data.
Often maximum likelihood training is used for this task, but in an operational
radar settings this method is too expensive.
Here, we treat the tuning parameters as unknown state, which should be inferred
from radar data.
As inference algorithm, we use a sequential importance resampling particle
filter :ref:`Sequential importance resampling (SIR) particle filter`.
Stochastic variogram matching is used as observation operator
:ref:`Observation operators`, while as stochastic uncertainty approximation
random walk is suitable, which is used as propagation model
:ref:`Propagation models`.
To mitigate the stochastic variogram, we approximate the observational
likelihood :ref:`Likelihoods` by a Laplace probability distribution.


Mathematical foundation
-----------------------
In inference, we want to estimate the state :math:`x_t` at time
:math:`t`, which is unknown.
An example for a state can be the current conditions of the atmosphere, the rain
rate at ground or tuning parameters for kriging.
The state cannot be observed without an error, instead we have a corrupted
(and partial) state observation at the same time,

.. math::

   y^o_t = h(x_t) + \epsilon^o_t,

   \epsilon^o_t \sim p(\epsilon^o).

Here, :math:`h(x)` is the known observation operator, transforming states into
observations.
:math:`\epsilon^o_t` is an additive and time-dependent observation error drawn
from a probability distribution :math:`p(\epsilon^o)`.
An example for observations can be radar reflectivities or rain gauge
measurements.
This observation operator can be recasted into a probabilistic model
:math:`p(y_t \mid x_t)`, where the observation is conditioned to the state at
same time. This probabilistic model measures the distance of the state to the
observations. This distance can be the mean-squared error for Gaussian
observation errors or the cross-entropy for Bernoulli distributed observation
errors.

As additional known model, we have a prognostic and dynamical model
:math:`m(x_{t-1})`, which propagates the state at time :math:`t-1` to time
:math:`t`.
This model can be a numerical weather prediction method, a nowcasting method
(see also :ref:`Nowcasting`) or the identity function, if we expect that
the state does not change over time. The initial state :math:`x_{t-1}` or the
model can have an uncertainty, leading to a time-dependent forecast error
:math:`\epsilon^b_t`.
If we assume that the forecast error is additive and drawn from a probability
distribution :math:`p(\epsilon^b)`, we can formulate this propagation step
similar to the observations,

.. math::

   x^b_t = m(x_{t-1}) + \epsilon^b_t,

   \epsilon^b_t \sim p(\epsilon^b).

Based on observations :math:`y_{1:t}` up to time :math:`t`, we want to calculate
the state :math:`x_t`.
We can estimate the conditional distribution of the state
:math:`p(x_t \mid y_{1:t})` with Bayes' theorem,

.. math::
    p(x_t \mid y_{1:t}) = \frac{p(y_t \mid x_t) p(x_t \mid y_{1:t-1})}
    {p(y_t \mid y_{1:t-1})}.

The conditional state is the likelihood of the observations
:math:`p(y_t \mid x_t)`, given the state, multiplied with prior state
:math:`p(x_t \mid y_{1:t-1})`, inferred based on all observation up to the
previous time :math:`t-1`.
The conditional probability of the current observation based on all observations
up to the previous time :math:`t-1` is a normalizing factor and can be often
neglected.
The prior state :math:`p(x_t \mid y_{1:t-1})` is given by propagation of the
previous state :math:`x_{t-1}` from :math:`t-1` to :math:`t`.
This propagation is formulated by the previously stated propagation equation and
includes the propagation model :math:`m(x)`.
This state likelihood given by previous observations draws the state estimate to
the prior state and can be seen as prior regularization of the solution.

This framework allows us to infer unknown parameters and states from radar data.
We describe in the following a method to directly solve Bayes' theorem with
probability distributions.

Particle filtering (sequential monte-carlo)
-------------------------------------------
One way to solve Bayes' theorem directly is to use monte-carlo methods, in form
of an ensemble of states or parameters.
Every observation time, ensemble members are weighted proportional to the
observational likelihood.
Ensemble members can be also resampled depending on their accumulated weights to
avoid a degenerated ensemble.
These steps are similar to evolutionary algorithms such that the fittest
ensemble members survive, while ensemble members with small weights die.

In the following, a special form of particle filtering called sequential
importance resampling particle filter are explained.

Sequential importance resampling (SIR) particle filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This special form of particle filter
(:py:class:`pylawr.transform.inference.SIRParticleFilter`) uses all previously
explained steps.
We will describe every step of the implementation here.

Let's start with a first guess ensemble of states or parameters
:math:`\boldsymbol{x}_0`.
If the propagation model has a stochastic random component, then every first
guess ensemble member can have the same value.
Further, we know nothing about the likelihood of different ensemble members such
that every :math:`i`-ensemble member has the same likelihood of
:math:`\frac{1}{n}` with :math:`n` as number of ensemble members.

As first step, every :math:`i`-th ensemble member is independently propagated to
the first observation time :math:`t=1` with propagation model
:math:`m(x^(i)_0)`.

In a second step, the observational likelihood :math:`p(y_1 \mid x^{b(i)}_1)`
based on propagated ensemble members is estimated.
Every propagated ensemble member is transformed into observational space, using
observation operator :math:`h(x^{b(i)}_1)`.
To apply Bayes' theorem, we have to multiply the observational likelihood by
prior ensemble weights.
The sum of ensemble weights is further normalized to unity.
Following this, every ensemble forecast has a weight, which is proportional to
the accumulated observational likelihood of this forecast.
This allows us to formulate the weight :math:`w^{(i)}_t` of the :math:`i`-th
ensemble at time :math:`t` in a recursive formula,

.. math::

    w^(i)_t = \frac{p(y_t \mid x^{b(i)}_t) w^{(i)}_{t-1}}
    {\sum^n_{j=1} {p(y_t \mid x^{b(j)}_t) w^{(j)}_{t-1}}},

    w^{(i)}_0 = \frac{1}{n}.

Estimated ensemble weights :math:`w^{(i)}_t` at time :math:`t` is the posterior
ensemble likelihood, which is used as prior likelihood at next observations
time.

The ensemble diverges over time, if we do not change the direction of forecast
runs.
This would lead to few ensemble members with very high likelihoods, while all
other ensemble members would have almost no weight.
To circumvent this problem, we have to add an additional step, called
resampling.
To measure divergence within the ensemble forecast, we can introduce the
effective number of ensemble members :math:`\hat{N}_{\text{eff}, t}` as inverse
of the sum of squared ensemble weights,

.. math::

    \hat{N}_{\text{eff}, t} = \frac{1}{\sum^n_{i=1}(w^{(i)}_t)^2}.

If the effective number of ensemble members is below a given threshold
:math:`T`, then we apply resampling.

We create a new ensemble in resampling by using propagated ensemble members and
corresponding ensemble weights.
New ensemble members are drawn from old ensemble with given posterior
likelihood.
Ensemble members with low likelihood are dismissed, while members with a high
likelihood are drawn more often.
The likelihood of the ensemble members is resetted to :math:`\frac{1}{n}`.
If no stochastic and random process is involved within the propagation model,
we need to randomly disturb the ensemble members to get an ensemble spread.

This new or old ensemble is then propagated up to next observation time, which
starts the cycle all over again.

The estimate of the state :math:`\tilde{x}^a_t` at time :math:`t` is the average
of the background ensemble, weighted by ensemble likelihood at the same time,

.. math::

    \tilde{x}^a_t = \sum^n_{i=1} w^{(i)}_t x^{b(i)}_t


We can say as summary that the SIR particle filter has three different steps,
which are cyclical repeated:

1. (Stochastic) **Propagation** of the prior ensemble
2. **Update** of the ensemble weights based on propagated ensemble,
    observations and ensemble weights from the previous step.
3. **Resampling** of the ensemble if a given threshold is reached

.. autosummary::
    pylawr.transform.inference.SIRParticleFilter


Implementations
---------------
In theory, this inference submodule can be used to infer different unknown
parameters and states.
One needs only to implement an observation operator, a propagation model and
the observational likelihood to start with inference.
Only implementations for adaptive kriging based on particle filters are
implemented at the moment.
These implementations are as simple as possible and can be seen as some type of
boilerplate to increase the possibilities for experimentation.

The interface for all operators is the same, as they need to be callable with
arbitrary keyword arguments.
Every inference algorithm has a ``fit``-method, where all additional keyword
arguments are passed to these functions.

In the following, we will explain the interface of the different functions and
further describe the implementations.

Observation operators
^^^^^^^^^^^^^^^^^^^^^
The first argument to the callable observation operator should be the state,
which is transformed into observation space.
The observations are passed as keyword argument ``obs``.
Every additional argument, e.g. the observational distance, has to be given as
arbitrary and additional keyword argument to the ``fit``-method of the inference
algorithm.

As only observation operator, a kernel variogram
(:py:func:`~pylawr.transform.inference.KernelVariogram`) is implemented.
A kernel is set within the observation operator, which is then evaluated as
theoretical variogram with given tuning parameters on given observation
distances (``obs_dist`` as additional keyword argument for ``fit`` in
inference).

.. autosummary::
    pylawr.transform.inference.KernelVariogram


Propagation models
^^^^^^^^^^^^^^^^^^
The first argument to the callable propagation model should be the state, which
is then propagated and changed in time.
The propagation model has to be currently stochastic, because the states are not
disturbed after resampling.

As only propagation model a Gaussian random walk
(:py:func:`~pylawr.transform.inference.random_walk`) is implemented.
Random walk takes an array and disturbs its values by truncated white noise,
which has a relative standard deviation (default = 5 %).
Random walk is suitable for processes, where only the increase of the
uncertainty is known, but the dynamics are unknown.
In this case the persistence as expectation is often a good candidate for a
dynamical model.

.. autosummary::
    pylawr.transform.inference.random_walk


Likelihoods
^^^^^^^^^^^
The first argument to the callable likelihoods should be the state.
This state is then compared to the observation, which are given as second
argument.

Two different likelihoods are specified at the moment:

* Gaussian likelihoods (:py:func:`~pylawr.transform.inference.gaussian_pdf`)
* Laplace likelihoods (:py:func:`~pylawr.transform.inference.laplace_pdf`)

Both likelihood functions take an additional variance (``var`` called) argument,
which can be used to specify the uncertainty of different observations.
This argument is specified as default to unity for observations with unknown
uncertainty.

.. autosummary::
    pylawr.transform.inference.gaussian_pdf
    pylawr.transform.inference.laplace_pdf


Functional API
--------------
At the moment, the functional API only supports inference for fitting of kriging
kernels.

Two different kinds of observation generation for variogram matching are
implemented. The first (:py:func:`~pylawr.functions.fit.sample_sq_diff`) samples
rain pixel pairs and squares their reflectivity deviations.
The second
(:py:func:`~pylawr.functions.fit.sample_variogram`) uses these sample pair to
estimate a stochastic variogram.
This stochastic variogram is used as observations in
:py:func:`~pylawr.functions.fit.fit_kriging`.
This function is operationally used to fit kriging instances to current radar
data based on SIR particle filter, variogram matching, random walk and laplacian
likelihood.
To infer tuning parameters for kriging from scratch, it is recommended to use
100 iterations or more.
An example on how to use this functional API to fit kriging is shown in
:ref:`Example for adaptive kriging with particle filter`.

.. autosummary::
    pylawr.functions.fit.sample_sq_diff
    pylawr.functions.fit.sample_variogram
    pylawr.functions.fit.fit_kriging