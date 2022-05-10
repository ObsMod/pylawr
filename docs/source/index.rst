pylawr: A Package For Weather Radar Processing
==============================================
The :py:mod:`pylawr` package is a Python_ package to load, process and plot
weather radar data. It is developed by the `Meteorological Institute of the
University of Hamburg <http://www.mi.uni-hamburg.de>`_ to process their local
area weather radars (LAWR). The LAWRs are used in a single or networked
environment. These single polarized X-band radars are used in combination with
nationwide C-band radar networks. The products are shown at the
`Wetterradar Homepage <http://wetterradar.uni-hamburg.de>`_.

.. _Python: https://www.python.org/

.. note:: The latest version of :py:mod:`pylawr` package is an alpha version, as
    well as this documentation, so please report any unstable behavior, code
    errors or misspelling. Thank you! Keep in mind the syntax can change
    drastically for increasing version/path number.

Read the following documentation to find solutions how to process radar data to
produce reliable and accurate products of our or possibly other weather radar
systems.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   start/overview
   start/installation
   examples/index

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   user/load-data
   user/field
   user/grid
   user/noise
   user/clutter
   user/remapping
   user/inference
   user/nowcasting
   user/attenuation
   user/plotting
   user/using-wradlib


.. toctree::
   :maxdepth: 1
   :caption: Help & References:

   api/pylawr
   appendix/internals
   appendix/references
