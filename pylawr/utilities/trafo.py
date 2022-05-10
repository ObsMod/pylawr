#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict


# Dictionary for the conversion from decibel to linear variables
from_decibel_to_linear = OrderedDict(
    dbz='z',
    dbrr='rr'
)


# Variable which can be used for the Z/R relationship
zr_vars = ['z', 'rr']