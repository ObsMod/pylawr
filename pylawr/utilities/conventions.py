#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict


naming_convention = OrderedDict(
    z={
        'units': 'mm**6/m**3',
        'standard_name': 'linear_equivalent_reflectivity_factor',
        'short_name': 'Z'
    },
    dbz={
        'units': 'dBZ',
        'standard_name': 'equivalent_reflectivity_factor',
        'short_name': 'dBZ'
    },
    rr={
        'units': 'mm/h',
        'standard_name': 'rainfall_rate',
        'short_name': 'RR'
    },
    dbrr={
        'units': 'dBR',
        'standard_name': 'decibel_rainfall_rate',
        'short_name': 'dBR'
    },
    pia={
        'units': 'dB',
        'long_name': 'path integrated attenuation',
        'short_name': 'PIA'
    },
    time={
        'standard_name': 'time'
    },
    azimuth={
        'standard_name': 'sensor_azimuth_angle',
        'units': 'degrees'
    },
    azimuth_offset={
        'long_name': 'offset sensor azimuth angle',
        'units': 'degrees'
    },
    range={
        'long_name': 'distance from sensor to center of each range gates '
                     'along the line of sight',
        'units': 'meters'
    },
    lat={
        'standard_name': 'latitude',
        'units': 'degrees_north'
    },
    lat_center={
        'standard_name': 'latitude',
        'comments': 'latitude of instrument location',
        'units': 'degrees_north'
    },
    lon={
        'standard_name': 'longitude',
        'units': 'degrees_east'
    },
    lon_center={
        'standard_name': 'longitude',
        'comments': 'longitude of instrument location',
        'units': 'degrees_east'
    },
    latitude={
        'standard_name': 'latitude',
        'units': 'degrees_north'
    },
    longitude={
        'standard_name': 'longitude',
        'units': 'degrees_east'
    },
    y={
        'long_name': 'y-coordinate in Cartesian system',
        'units': 'meters'
    },
    x={
        'long_name': 'x-coordinate in Cartesian system',
        'units': 'meters'
    },
    zsl={
        'standard_name': 'altitude',
        'units': 'meters'
    },
    zsl_center={
        'standard_name': 'altitude',
        'comments': 'altitude of instrument above mean sea level',
        'units': 'meters'
    },
    ele={
        'long_name': 'sensor elevation angle',
        'units': 'degrees'
    },
    noise_level={
        'long_name': 'noise level',
        'units': 'mm**6/m**3/r**2'
    },
)
