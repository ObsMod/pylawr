# pylawr

[![CI](https://github.com/ObsMod/pylawr/actions/workflows/ci.yml/badge.svg)](https://github.com/ObsMod/pylawr/actions/workflows/ci.yml)

The `pylawr` package is a Python package to load, process and plot weather radar data. It is developed by the Meteorological Institute of the University of Hamburg to handle measurement data of their local area weather radars (LAWR). These low-cost, operational, single-polarized X-band weather radars provide measurements with high temporal (30 s), range (60 m), and sampling (1°) resolution within a 20 km scan radius refining observations of the German nationwide C-band radars.

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

* Finn Burgemeister
* Tobias Sebastian Finn
* Yann Büchau
* Maximilian Schaper
