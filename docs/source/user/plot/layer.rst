The :any:`pylawr.plot.layer` contains several different layer types.
They are all based on the :py:class:`~pylawr.plot.layer.BaseLayer`.
An example for the use of the different layers together with the
:py:class:`~pylawr.plot.Plotter` is given at the end.

Base layer
^^^^^^^^^^

The :py:class:`~pylawr.plot.layer.BaseLayer` introduces the
default settings for all layers. The settings contain plot options as well as
some additional parameters specifying the aspects and specific settigns
(if necessary).

Some of those parameter are used as kwarguments for specific plot functions.
Those parameter are specified by the name of the plot function
for example *pcolormesh*.

More settings can be added or changed by calling the *settings*
attribute of any given layer.

**NOTE** if one wants to restore the default
settings, the *settings* attribute has to be set to ``None``.


Header layer
^^^^^^^^^^^^

The :py:class:`~pylawr.plot.layer.LawrHeaderLayer`
is used to create the default header. It contains an additional
attribute which is
:py:attr:`~pylawr.plot.layer.LawrHeaderLayer.header_info`.
This attribute contains the information that will be displayed within
the header. There are three keys necessary for this dictionary:

- ``title`` containg the title string
- ``left`` containg a dictionary with *key/value* pairs which wil be displayed
  each within a seperate line on the lower left side
- ``right`` containg a dictionary with *key/value* pairs which wil be displayed
  each within a seperate line on the lower right side

The default
:py:attr:`~pylawr.plot.layer.LawrHeaderLayer.header_info`
contains some placeholder texts.

Background layer
^^^^^^^^^^^^^^^^

This :py:class:`~pylawr.plot.layer.BackgroundLayer` adds
a background map to an axis with a projection.
The map is based on :py:class:`~cartopy.io.img_tiles.OSM`.
A default resolution is specified within the
:py:attr:`~pylawr.plot.layer.BaseLayer.settings` attibute.

**MISSING** an initialisation stage, which is not jet implemented.

Radar field layer
^^^^^^^^^^^^^^^^^

The :py:class:`~pylawr.plot.layer.RadarFieldLayer` can be used to
plot a given data onto an axis, which should have a projection. The given data
has to be either gridded (for example :py:class:`~pylawr.RadarField` )
or an additional grid has to be provided for the given data.

The plot-object resulting out of the call to *pcolormesh* is stored within the
*plot_stroe* attribute of the
:py:class:`~pylawr.plot.layer.RadarFieldLayer`.
Thereby it is possible to reference the plot-object for the creation
of a colorbar.

Colorbar layer
^^^^^^^^^^^^^^

The :py:class:`~pylawr.plot.layer.ColorbarLayer` creates a colorbar
matching a given :py:class:`~pylawr.plot.layer.RadarFieldLayer`.
The reference to the corresponding plot-object handle will be made internally.

The positioning of the colorbar is adjusted by some parameters set within the
:py:attr:`~pylawr.plot.layer.BaseLayer.settings` attibute.

**NOTE** the only important thing is, that the
:py:class:`~pylawr.plot.layer.ColorbarLayer` is plotted after the
corresponding :py:class:`~pylawr.plot.layer.RadarFieldLayer`.
So that the plot-object reference does exist.

Removing a layer
^^^^^^^^^^^^^^^^

If one wants to remove a specific *layer* after the plot is plotted,
it is possible via the *layer* object.
All *layers* posess a :py:meth:`~pylawr.plot.layer.BaseLayer.remove`
method.
This method will remove all parts of the *layer* that are added to the figure.



