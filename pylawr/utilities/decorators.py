#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import functools
import time
import inspect
import collections.abc
import six

# External modules

# Internal modules


logger = logging.getLogger(__name__)


def log_decorator(fn_logger):
    def log_decorator_decoration(fn):
        """
        Decorator that measures the time between start and end of the function.
        The start is logged as info message, while the end is logged as info
        message with function duration.

        Parameters
        ----------
        fn : python func
            This function is wrapped by this property.

        Returns
        -------
        fn_log : python func
            The wrapped function, which logs start and end.
        """
        @functools.wraps(fn)
        def fn_log(*args, **kwargs):
            log_start = time.time()
            tmp_logger = fn_logger.getChild(fn.__name__)
            tmp_logger.info("Started")
            return_vals = fn(*args, **kwargs)
            duration = time.time() - log_start
            end_str = "Ended, duration: {0:.3f} s".format(
                duration
            )
            tmp_logger.info(end_str)
            return return_vals
        return fn_log
    return log_decorator_decoration


def lazy_property(fn):
    """
    Decorator that makes a property lazy-evaluated.
    Based on: http://stevenloria.com/lazy-evaluated-properties-in-python/
    """
    attr_name = '_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


def tuplesetter(prop, len_tuple=2, valid_types=None):
    """
    This decorator can be used to make sure that the set attribute is a tuple.
    The decorated method needs to return the value, which should be set. If the
    value is an iterable, it will be converted into a tuple and shortened to
    the given tuple length starting from the beginning. If the value is None,
    then the attribute will be set to None. For any other type the value is
    converted into a tuple with `len_tuple` times the value as elements.

    Parameters
    ----------
    prop : property
        the property
    len_tuple : int
        The length of the set tuple. If the return value of the decorated method
        is an iterable it will be shortened to this length. Default is 2.
    valid_types : python type, tuple(python types) or None
        The tuple elements will be tested against these valid types with
        :py:func:``isinstance``. If this is None, the test will be skipped.
        Default is None.

    Returns
    -------
    callable
        The decorated callable
    """
    def tuplesetter_decorator(fn):
        propname = fn.__name__
        attrname = "_{}".format(propname)
        fn_args = inspect.getfullargspec(fn).args

        if len(fn_args) != 2:
            raise ValueError(
                "`tuplesetter` decorator can only be used for methods that "
                "take the object reference as first argument and the new "
                "property value as second argument!"
            )
        elif fn_args[0] != 'self':
            raise ValueError(
                "`tuplesetter` decorator can only be used for methods that "
                "take the object reference as first argument, which needs "
                "to be called `self`!"
            )

        def setter(self, newval):
            is_iterable = isinstance(newval, collections.abc.Iterable) and \
                          not isinstance(newval, six.string_types)
            if is_iterable and len(newval) < len_tuple:
                raise ValueError(
                    "The given new value is too short! "
                    "desired length: {0:d}, actual length: {1:d}".format(
                        len_tuple, len(newval)
                    )
                )
            elif is_iterable:
                new_tuple = tuple(newval[:len_tuple])
            elif newval is None:
                new_tuple = None
            else:
                new_tuple = (newval,)*len_tuple

            if valid_types is not None and \
                    not all([isinstance(e, valid_types) for e in new_tuple]):
                raise TypeError(
                    "Not all elements of the given tuple have the right type!"
                )
            converted = fn(self, new_tuple)
            setattr(self, attrname, converted)
        setter.__doc__ = fn.__doc__
        setter = prop.setter(setter)
        return setter

    return tuplesetter_decorator
