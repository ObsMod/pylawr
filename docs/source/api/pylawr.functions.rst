pylawr.functions
================

.. toctree::
    :maxdepth: 1

    pylawr.functions.fit

.. autosummary::
    pylawr.functions.fit.extrapolate_offline
    pylawr.functions.fit.fit_extrapolator
    pylawr.functions.fit.fit_kriging
    pylawr.functions.fit.sample_sq_diff
    pylawr.functions.fit.sample_variogram

.. toctree::
    :maxdepth: 1

    pylawr.functions.grid

.. autosummary::
    pylawr.functions.grid.get_cartesian
    pylawr.functions.grid.get_latlon_grid
    pylawr.functions.grid.get_masked_grid
    pylawr.functions.grid.prepare_grid
    pylawr.functions.grid.remap_data

.. toctree::
    :maxdepth: 1

    pylawr.functions.input

.. autosummary::
    pylawr.functions.input.read_dwd_hdf5
    pylawr.functions.input.read_lawr_ascii
    pylawr.functions.input.read_lawr_nc_level0
    pylawr.functions.input.read_lawr_nc_new
    pylawr.functions.input.read_lawr_nc_old

.. toctree::
    :maxdepth: 1

    pylawr.functions.output

.. autosummary::
    pylawr.functions.output.save_netcdf

.. toctree::
    :maxdepth: 1

    pylawr.functions.plot

.. autosummary::
    pylawr.functions.plot.create_default_plotter
    pylawr.functions.plot.plot_rain_clutter
    pylawr.functions.plot.plot_rain_rate
    pylawr.functions.plot.plot_reflectivity

.. toctree::
    :maxdepth: 1

    pylawr.functions.transform

.. autosummary::
    pylawr.functions.transform.correct_attenuation
    pylawr.functions.transform.correct_attenuation_dual
    pylawr.functions.transform.correct_attenuation_dwd
    pylawr.functions.transform.correct_attenuation_lawr
    pylawr.functions.transform.extrapolation_offline
    pylawr.functions.transform.interpolate_missing
    pylawr.functions.transform.remove_clutter_dwd
    pylawr.functions.transform.remove_clutter_lawr
    pylawr.functions.transform.remove_noise
