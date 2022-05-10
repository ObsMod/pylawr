Internals
=========

This page is dedicated for internal usage of the software package. There you
find some development plans, coding style guide and conventions.

Conventions
-----------

There are some conventions, e.g. naming conventions. The internals of this
processing software are based on these conventions. Follow these conventions
if you want to contribute to this software.

Coding Conventions
^^^^^^^^^^^^^^^^^^
* PEP8
* Numpy docstring
* Maximum line length: 80
* At least one test per object
* Pass all tests before pull request
* Check and update requirements and environment.yml

Testing Conventions
^^^^^^^^^^^^^^^^^^^
* Try to test every line of code
* Every function and object should have a docstring describing what happens within the call.
* The test-driven development paradigm is desirable


Naming Conventions
^^^^^^^^^^^^^^^^^^

It is possible to set the DataArray name to the radar naming conventions, which
are used in this package. The conventions are based on OPERA_, NCAR_,
`CF Conventions <https://cfconventions.org>`_. We try to follow
the product standard defined by :cite:`lammert_andrea_2018`.
At the moment, following different variables are implemented:

- 'dbz' (:math:`\mathrm{dBZ}`): reflectivity in decibel (standard_name: equivalent_reflectivity_factor, short_name: dBZ)
- 'z' (:math:`\mathrm{mm^6\,m^{-3}}`): linear reflectivity (standard_name: linear_equivalent_reflectivity_factor, short_name: Z)
- 'rr' (:math:`\mathrm{mm\,h^{-1}}`): linear rain rate (standard_name: rainfall_rate, short_name: RR)
- 'dbrr' (:math:`\mathrm{dBR}`): rain rate in decibel (standard_name: decibel_rainfall_rate, short_name: dBR)
- 'pia' (:math:`\mathrm{dB}`): path integrated attenuation (long_name: path integrated attenuation, short_name: PIA)
- 'time' (): time (standard_name: time)
- 'azimuth' (:math:`^\circ{}`): azimuth angle (standard_name: sensor_azimuth_angle)
- 'azimuth_offset' (:math:`^\circ{}`): offset of azimuth angle (long_name: offset sensor azimuth angle)
- 'range' (:math:`\mathrm{m}`): range (long_name: distance from sensor to center of each range gates along the line of sight)
- 'lat' (:math:`^\circ{}\,\mathrm{N}`): latitude (standard_name: latitude)
- 'lat_center' (:math:`^\circ{}\,\mathrm{N}`): latitude of instrument location (standard_name: latitude)
- 'lon' (:math:`^\circ{}\,\mathrm{E}`): longitude (standard_name: longitude)
- 'lon_center' (:math:`^\circ{}\,\mathrm{E}`): longitude of instrument location (standard_name: longitude)
- 'latitude' (:math:`^\circ{}\,\mathrm{N}`): latitude (standard_name: latitude)
- 'longitude' (:math:`^\circ{}\,\mathrm{E}`): longitude (standard_name: longitude)
- 'y' (:math:`\mathrm{m}`): Cartesian y-coordinate (long_name: y-coordinate in Cartesian system)
- 'x' (:math:`\mathrm{m}`): Cartesian x-coordinate (long_name: x-coordinate in Cartesian system)
- 'zsl' (:math:`\mathrm{m}`): altitude / height above mean sea level (standard_name: altitude)
- 'zsl_center' (:math:`\mathrm{m}`): altitude of instrument above mean sea level (standard_name: altitude)
- 'ele' (:math:`^\circ{}`): beam elevation angle (long_name: 'sensor elevation angle')
- 'noise_level' (:math:`\mathrm{mm^6\,m^{-3}\,m^{-2}}`): noise level (long_name: 'noise_level')

The variable can be set with: ``array.lawr.set_variable(var_name)``, where
``var_name`` is the name of the variable. If this method is used, then the
name, short name, long name and unit will be set as defined above.

If the data is read with a DataHandler, then all variables will be set
automatically as defined the conventions.

.. _OPERA: https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwi8gvbp-tnYAhWBKewKHeUxCggQFggsMAA&url=http%3A%2F%2Feumetnet.eu%2Fwp-content%2Fuploads%2F2017%2F01%2FOPERA_hdf_description_2014.pdf&usg=AOvVaw1-VbAclpxUs1Llrrf5RxMz
.. _NCAR: https://github.com/NCAR/CfRadial


How can I add a new radar variable?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The naming conventions with short name, long name, unit and standard name are
defined within `pylawr/utilities/conventions` as `OrderedDict` with the variable
name `naming_convention`. There you can add a new variable, which needs to have
the same structure as the other variables.

Additionally, you need to add a way how your variable can be converted to linear
reflectivity. For this you need to add your standard variable name as key to the
`from_dict` within `pylawr/field.py`. The value need to be a valid way how your
variable can be converted to linear reflectivity.

There exist within the `field` class an utility method `_convert_field` which
takes a conversion function and a target variable name as arguments. Also, you
can use the `func_compose` function within `pylawr/utilities/utilities.py`,
which allows you to compose different functions together.

The easiest way to add the route from and to linear reflectivity would be to add
an additional method to the `Field` accessor with an additional `inverse`
keyword, an example for this is the `zr_convert` method within the `Field`
accessor. It is recommended to add tests to `tests/unit_tests/test_field.py`
to check if your conversion implementation is valid.
