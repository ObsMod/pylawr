Installation
============

The :py:mod:`pylawr` package is a Python package with several dependencies to
libraries, which need to be installed additionally. A installed Python
interpreter is required. Just follow the instructions below.

Conda
-----
The easiest way to get everything installed is to use conda_ with the provided
yml-file and install the package with pip.

.. _conda: http://conda.io/

.. code-block:: none

    conda env create -f environment.yml
    source activate pylawr
    pip install ./path_to_pylawr_package
    source deactivate pylawr

pip
---
If you don't use conda, install the package with pip based on the requirements,
but note there could be more dependencies, the python-packages in
`requirements.txt` require.

.. code-block:: none

    pip install -r requirements.txt

Development
-----------
To develop :py:mod:`pylawr` use conda and install the package with
pip in
editable mode.

.. code-block:: none

    conda env create -f dev_environment.yml
    source activate pylawr
    pip install -e ./path_to_pylawr_package
    source deactivate pylawr

Requirements
------------
The following block lists the required dependencies.

.. code-block:: none

    cartopy
    coverage
    gdal
    geos
    hdf5
    ipython
    json-c
    libnetcdf
    matplotlib
    netcdf4
    paramiko
    pip
    poppler
    proj4
    pyproj
    python
    pytz
    pyyaml
    requests
    setuptools
    scikit-image
    scikit-learn
    wradlib
    xarray
    numpy
    scipy