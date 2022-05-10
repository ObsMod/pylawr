#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import io
import datetime
import pytz

# External modules
import numpy as np
import xarray as xr

# Internal modules
from pylawr.datahandler.base import DataHandler
from pylawr.utilities.conventions import naming_convention


logger = logging.getLogger(__name__)


class LawrHandler(DataHandler):
    """
    The LawrHandler is constructed to read in LawrText files.

    Parameters
    ----------
    fh : filelike object
        The data is read from this filelike object.
    """
    def __init__(self, fh):
        super().__init__(fh=fh)
        self._header = None
        self._file = None

    @property
    def file(self):
        if self._file is None:
            data = self._fh.read()
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            self._file = io.StringIO(data)
            try:
                self._fh.seek(0)
            except io.UnsupportedOperation:
                # Seek is not supported with given file handler
                pass
        return self._file

    @property
    def header(self):
        """
        Get the decoded header of the file as dict.
        """
        if self._header is None:
            self.file.seek(0)
            raw_header = self.file.readline()
            self._header = self._decode_header(raw_header)
            self.file.seek(0)
        return self._header

    def _get_available_dates(self):
        """
        Get a list of the available dates as datetime.datetime.
        """
        return [self.header['datetime'], ]

    def _decode_header(self, header):
        """
        Decode the given header line into a lawr header.

        Parameters
        ----------
        header : str
            This header line is decoded.

        Returns
        -------
        decoded_header : dict(str, value)
            The decoded header dict.
        """
        prepared_header = self._prepare_header(header)
        decoded_header = dict(
            radartype=prepared_header[0],
            datetime=self._decode_datetime(*prepared_header[1:3]),
            datestr=prepared_header[1],
            timezone=prepared_header[2])
        decoded_header.update(self._decode_keyval(prepared_header[3:]))
        return decoded_header

    @staticmethod
    def _prepare_header(header):
        """
        Clean the header string and split the header string.

        Parameters
        ----------
        header : str
            This header is cleaned and splitted.

        Returns
        -------
        prepared_header : list
            The cleaned and splitted substrings of the header.
        """
        prepared_header = header.split()
        return prepared_header

    @staticmethod
    def _decode_datetime(dt_str, tz_str=None):
        """
        Decode a given datetime string to a datetime object in UTC.

        Parameters
        ----------
        dt_str : str
            This datetime string is decoded and transformed to a datetime
            object. The format should be %y%m%d%H%M%S.
        tz_str : str or None, optional
            The timezone information string. The timezone is converted to pytz
            timezone. If the timezone is  None, the timezone will not be set.
            Default is None.

        Returns
        -------
        decoded_dt : datetime.datetime
            The decoded datetime object. If a timzone string was given, the
            object will be localized and returned in UTC without
            timezone information.
        """
        decoded_dt = datetime.datetime.strptime(dt_str, '%y%m%d%H%M%S')
        if tz_str:
            tzinfo = pytz.timezone(tz_str)
            decoded_dt = tzinfo.localize(decoded_dt)
            decoded_dt = decoded_dt.astimezone(pytz.utc)
            decoded_dt = decoded_dt.replace(tzinfo=None)
        return decoded_dt

    @staticmethod
    def _decode_keyval(header_list):
        """
        A key value dict will be created by given list. Equal signs are
        searched, the list value before the sign is used as key and the value
        after the sign is used as value.

        Parameters
        ----------
        header_list : list(str)
            The equal sign are searched within this list.

        Returns
        -------
        decoded_keyval : dict(str, float)
            The decoded key-value dictionary. If the dict is empty there was
            no key value pair found.
        """
        decoded_keyval = {
            header_list[k-1]: float(header_list[k+1])
            for k, sub in enumerate(header_list)
            if sub == '=' and k > 0 and k+1 < len(header_list)
        }
        return decoded_keyval

    def _decode_data(self, data_lines):
        """
        Decode the given data lines into a numpy array.

        Parameters
        ----------
        data_lines : list(str)
            The data lines which should be decoded.

        Returns
        -------
        decoded_array : numpy.ndarray
            The decoded data array. The first dimension are the azimuth angles,
            the second dimension is the range.
        """
        decoded_lines = [self._prepare_dataline(l)[1:]
                         for l in data_lines if l.startswith('ppw')]
        decoded_array = np.array(decoded_lines, dtype=np.float32)
        return decoded_array

    @staticmethod
    def _prepare_dataline(data_line):
        """
        Clean and split the given data line.

        Parameters
        ----------
        data_line : str
            This data line is cleaned and split.

        Returns
        -------
        prepared_line : list
            The cleaned and split data line.
        """
        prepared_line = data_line.split()
        return prepared_line

    def get_reflectivity(self):
        """
        Get the data from the file handler.

        Returns
        -------
        radar_field : xr.DataArray
            The decoded radar field.
        """
        self.file.seek(0)
        raw_header = self.file.readline()
        raw_data = self.file.readlines()
        try:
            self._header = decoded_header = self._decode_header(raw_header)
            decoded_data = self._decode_data(raw_data)
            azi_len, range_len = decoded_data.shape
        except ValueError:
            raise ValueError('The given file is corrupted and couldn\'t '
                             'be decoded!')
        if azi_len != decoded_header['n_p']:
            missing_angles = int(decoded_header['n_p'] - azi_len)
            raise ValueError(
                'There are {0:d} missing azimuth angles within the '
                'file!'.format(missing_angles)
            )
        # Decoded header is already UTC
        utc_date = decoded_header['datetime'].replace(tzinfo=None)
        coordinates = {
            'time': (('time',),
                     np.array((utc_date, ), dtype='datetime64[ns]'),
                     naming_convention['time']),
            'azimuth': (('azimuth',),
                        np.arange(azi_len),
                        naming_convention['azimuth']),
            'range': (('range',),
                      np.arange(range_len),
                      naming_convention['range']),
        }
        attrs = dict(unit='dBZ')
        attrs.update({k: decoded_header[k] for k in decoded_header.keys()
                      if k != 'datetime'})
        radar_field = xr.DataArray(
            data=decoded_data[np.newaxis, ...],
            coords=coordinates,
            dims=['time', 'azimuth', 'range'],
            attrs=attrs,
        )
        radar_field = radar_field.lawr.set_variable('dbz')
        self.file.seek(0)
        return radar_field
