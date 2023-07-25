Overview
========
Hamburg's python package for weather radar processing - :py:mod:`pylawr` - is a
product of own algorithms and other products. The goal is to produce reliable
and accurate estimates of precipitation based on noisy and cluttered
radar measurements.

Measurements
------------
The origin of this package is the project Precipitation and Attenuation
Estimates from a High-Resolution Weather Radar Network (PATTERN).
For now, the Meteorological Institute of the Universität Hamburg operates two
low-cost local area weather radars (LAWR) in the region of Hamburg. These single
polarized X-band radars observing precipitation with following specifications
:cite:`lengfeld2014`.

==============================  ==============
Performance parameters          Specifications
==============================  ==============
Range resolution                60 m
Time resolution                 30 s
Angular resolution              2.8 deg
Sampling resolution in azimuth  1 deg
Maximum range                   20 km
Calibration accuracy            +-1 dB
Frequency                       9410 MHz
Beam width                      2.8 deg
==============================  ==============

User Qualifications
-------------------
We defined three example users with different qualifications, who should be able
to use the python package and took it into the development:

* **low-level-api** (Experienced Programmer):
    You need no description, because you are able to use and expand the
    :py:mod:`pylawr` package.

* **functional-api** (Applied User):
    You are able to create easy python scripts. We provide extended functions to
    use for important applications (e.g. read data to get the radar
    reflectivity, remove clutter, estimate the noise level, plot rain field,
    etc.).

* **script-api** (Beginner):
    You are able to start a python script. We may provide scripts to process
    radar data, e.g. to process the actual radar measurements.

Directory Structure
-------------------
* **docs**:
    The docs folder contains the files to build the documentation for this
    project.

* **examples**:
    The examples folder contains some examples to show what could be done with
    the package.

* **pylawr**:
    The :py:mod:`pylawr` folder contains the proper modules.

* **tests**:
    The tests folder contains the tests for the different submodules.

List Of Abbreviations
---------------------

* **LAWR** local area weather radar
* **WRX** X-band weather radar
* **WRC** C-band weather radar
* **MRR** Micro rain radar
* **HHG** LAWR/WRX located on the Geomatikum in Hamburg.
* **ALT** LAWR/WRX located in Jork (Altes Land) in the southwest of Hamburg.

Why pylawr?
-----------
The benefits and goals of :py:mod:`pylawr` package are:

* Post- and operational processing of radar measurements in real time
* Flexible plotting of weather radar data
* Combined processed products of several X-Band radars with C-Band radar