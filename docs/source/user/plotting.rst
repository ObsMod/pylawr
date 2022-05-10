Plotting
========
This Section will cover all the details neccessary to use the plotting utilities
of the :py:mod:`pylawr` module. In addition to the described options, every aspect
of a plot might be adjusted via unmentioned alteration of the default settings.
Plotting in splitted up into two main parts – plotter and layers.

The plotter is used as main object to handle a complete plot with its figure and
subplots.
Subplots are a part of a whole plot, which can be independently controlled from
other subplots.
Subplots are often implicitly used and created.
Subplots are a composition of different layers, describing what is shown on a
given subplot.

Functionalities of plotter and layers are explained in the following.

Plotter
-------

.. include:: plot/plotter.rst

.. autosummary::
    pylawr.plot.Plotter
    pylawr.plot.Subplot

Layers
------

.. include:: plot/layer.rst

.. autosummary::
    pylawr.plot.layer.BaseLayer
    pylawr.plot.layer.BackgroundLayer
    pylawr.plot.layer.ColorbarLayer
    pylawr.plot.layer.LawrHeaderLayer
    pylawr.plot.layer.RadarFieldLayer

Funtional API
-------------
The functional api can be used to plot radar data on a default plot setup.
It is possible to create a plotter with default plot settings
(:py:func:`~pylawr.functions.plot.create_default_plotter`), which are
operationally used to create radar images.
This function is also used as basis function for other plotting functions in
this api.
The raw reflectivity data can be plotted with default header and colorbar
information (:py:func:`~pylawr.functions.plot.plot_reflectivity`).
The rain rate can be similarly plotted
(:py:func:`~pylawr.functions.plot.plot_rain_rate`).
Further the rain rate and a clutter map can be combined with two different
color maps to show results from clutter detection
(:py:func:`~pylawr.functions.plot.plot_rain_clutter`).

.. autosummary::
    pylawr.functions.plot.create_default_plotter
    pylawr.functions.plot.plot_reflectivity
    pylawr.functions.plot.plot_rain_rate
    pylawr.functions.plot.plot_rain_clutter

Usage Example
-------------

This example shows the most general usage of the
:py:class:`~pylawr.plot.Plotter` and the :any:`pylawr.plot.layer`.

Default figure consists of three axis ('header', 'map', 'colorbar').

.. code-block:: python

    # define plot areas
    default_gridspec_slices = {
        'header': [
            slice(2),
            slice(None),
        ],
        'map': [
            slice(2, None),
            slice(12),
        ],
        'colorbar': [
            slice(2, None),
            slice(12, None),
        ],
    }

    # instantiate `Plotter`
    plotter = Plotter(grid_size=(14, 14), grid_slices=default_gridspec_slices,
                      backend_name='agg', figsize=(13, 9))

    map_subplot = plotter.subplots.get('map')
    rotated_pole = ccrs.RotatedPole(-170.415, 36.063)
    map_subplot.projection = rotated_pole
    map_subplot.auto_extent = True
    map_subplot.extent = dict(
        lon_min=9.6,
        lon_max=10.3,
        lat_min=53.35,
        lat_max=53.76
    )

    # define `RadarFieldLayer` to plot the radar field
    rain_layer = RadarFieldLayer(radar_field=rate, zorder=1)
    rain_layer['cmap'] = available_cmaps['rain']
    rain_layer['cmap'].set_bad((0, 0, 0, 0))
    rain_layer['norm'] = colors.LogNorm(vmin=0.1, vmax=200)

    # define `LawrHeaderLayer` to add an header to the figure
    hl = LawrHeaderLayer()
    plot_date = rate.indexes['time'][-1].tz_localize('UTC')
    current_date = plot_date.tz_convert('Europe/Berlin').strftime('%Y-%m-%d')
    current_time = plot_date.tz_convert('Europe/Berlin').strftime('%H:%M:%S')
    hl.left = OrderedDict(Datum=current_date, Zeit=current_time)
    hl.title = title
    hl.right = OrderedDict(Radar=radar, Parameter='Regenrate [mm/h]')

    # define `ColorbarLayer`
    cl = ColorbarLayer(rain_layer)

    # add the layers to the plotter and plot
    plotter.add_layer('map', rain_layer)
    plotter.add_layer('colorbar', cl)
    plotter.add_layer('header', hl)
    plotter.plot()

    # edit the colorbar
    cl.colorbar.set_ticks([0.1, 0.5, 1, 2, 5, 10, 100])
    cl.colorbar.ax.set_yticklabels(
        ['0.1', '0.5', '1', '2', '5', '10', '>100'])

    plotter.show()
