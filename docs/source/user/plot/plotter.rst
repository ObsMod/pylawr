The :py:class:`~pylawr.plot.Plotter` exists to simplify the composition and
use of any :any:`pylawr.plot.layer`. As such it allows for the handling of any
:any:`matplotlib.pyplot.figure` or :any:`matplotlib.pyplot.axis`
related setting.

Figure structure
^^^^^^^^^^^^^^^^

The composition of the figure is defined upon initialization of the
:class:`~pylawr.plot.Plotter` object, but can be changed later on.
Changing the figure composition is done by using
:any:`matplotlib.gridspec`
via the :py:attr:`~pylawr.plot.Plotter.gridspec` and
:py:attr:`~pylawr.plot.plotter.gridspec_slices` attributes
(and respective initialisation parameters).

After changing the composition or the **projection** of a axis was changed, one
has to reset the :py:attr:`~pylawr.plot.Plotter.ax_dict`
to create a new figure and ax-handles. This is done by calling the
:py:meth:`~pylawr.plot.Plotter.setup_new_axis_dict`.
If only a specific axis was changed the
:py:meth:`~pylawr.plot.Plotter.setup_axis`
method can be used.

A few figure parameters can be set via the
:py:attr:`~pylawr.plot.Plotter.ax_settings`.

Ax_settings attribute
^^^^^^^^^^^^^^^^^^^^^

Additional parameters during the initilalisation of the
:py:class:`~pylawr.plot.Plotter` will be added to
:py:attr:`~pylawr.plot.Plotter.ax_settings`.
Some of them are used to change the figure. The possible keys are:

- **tick_params** setting axis ticks for all axis (default= everything off)
- **set_frame_on** setting the frame for all axis (default= True)
- **figsize** specifying the figure size (default= (13, 9))
- **extent_auto_expand** special setting for the ''map'' layer (see section map)
  (default= True)
- **projections** containing the *ax_specifier* as key with respective
  projection (default= empty dict)
- **extent_projection** specifying the projection of the used data
  (default :any:`cartopy.crs.PlateCarree`)

**NOTE** additional keys will be ignored.

**NOTE** entries will only be added to
:py:attr:`~pylawr.plot.Plotter.ax_settings`.
In order to reset the settings to the default ones, one has to use:
.. code-block:: python

    plotter.ax_settings = None

Handling axes
^^^^^^^^^^^^^

The :any:`matplotlib.pyplot.axis` are defined by the
:py:attr:`~pylawr.plot.Plotter.gridspec_slices`.
Therefore the have also a name. This name is refered to as the **ax_specifier**.
The *ax_specifier* is necessary to manipulate a specific axis after its
creation. To acess the respective :any:`matplotlib.pyplot.axis` one can use the
:py:meth:`~pylawr.plot.Plotter.get_ax`.

adding a projection
    Adding a projection to an axis is done by utilizing :any:`cartopy`.
    By calling the :py:meth:`~pylawr.plot.Plotter.add_projection_to_ax`
    method it is possible
    to add a given projection to the specified axis. The
    :py:attr:`~pylawr.plot.Plotter.axis_dict`
    will be automatically reset to account for
    the changed projection.

setting the extent of an axis
    This is only necessary for axis with a projection. It is done by utilizing
    the :py:meth:`~pylawr.plot.Plotter.set_ax_extent` method after the axis is
    created. Due
    to the fact that axes
    with a projection are always set to an equal aspect ratio (to display the
    projection correctly), the figure composition gets distored. To counter this
    effect the *extent_auto_expand* setting (within
    :py:attr:`~pylawr.plot.Plotter.ax_settings`)
    is introduced. If requested the given extent will be *expanded* to ensure
    that the composition of the subplots within the figure does not change.

Handling layers
^^^^^^^^^^^^^^^

A :any:`pylawr.plot.layer` can be added to a :py:class:`pylawr.plot.Plotter`
by calling the :py:meth:`~pylawr.plot.Plotter.add_layer` method.

A specific layer can be changed by calling the
:py:meth:`~pylawr.plot.Plotter.swap_layer` method. In order to swap a layer
with an already existing layer, one of three parameter has to be given.
Either the old *layer* object, the index of the layer within the *layer_list*
of the *plotter*, or the corresponding *zorder* which shall be redone.
The specified layer will be swapped and if the zorder of the new layer is not
specified it will be set to match the old one.

If the *zorder* parameter is used and two old layers happen to have
*zorder* that shall be replaced, both existing layers will be removed.