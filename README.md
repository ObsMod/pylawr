# pylawr: A Python Package For Processing Local Area Weather Radars

[![CI](https://github.com/ObsMod/pylawr/actions/workflows/ci.yml/badge.svg)](https://github.com/ObsMod/pylawr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ObsMod/pylawr/branch/main/graph/badge.svg)](https://codecov.io/gh/ObsMod/pylawr)
[![Documentation Status](https://readthedocs.org/projects/pylawr/badge/?version=latest)](https://pylawr.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8182628.svg)](https://doi.org/10.5281/zenodo.8182628)

The `pylawr` package is a Python package to load, process and plot weather radar data. It is developed by the Meteorological Institute of the University of Hamburg to handle measurement data of their local area weather radars (LAWR). These low-cost, operational, single-polarized X-band weather radars provide measurements with high temporal (30 s), range (60 m), and sampling (1°) resolution within a 20 km scan radius refining observations of the German nationwide C-band radars.

## Documentation

Learn more about `pylawr` and find code examples in its documentation at <https://pylawr.readthedocs.io>. 

## Prerequisites

This project is programmed with python 3.

Installation of prerequisites with a conda virtualenv.
```
conda env create -f environment.yml
```

Installation of prerequisites with pip
```
pip install -r requirements.txt
```

## Authors

* [Finn Burgemeister](https://github.com/fiburg)
* [Tobias Sebastian Finn](https://github.com/tobifinn)
* [Maximilian Schaper](https://github.com/fontibon)
* [Yann Büchau](https://github.com/nobodyinperson)
