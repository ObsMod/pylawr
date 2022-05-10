#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
from functools import partial
import abc
import itertools

# External modules
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


class BaseKernel(np.lib.mixins.NDArrayOperatorsMixin):
    """
    This BaseKernel is a base object for all Kernel classes. This BaseKernel
    defines all available mathematical operation for any kernel. Many different
    `numpy ufuncs` can be used to manipulate this kernel, resulting in a new
    kernel.
    """
    __array_priority__ = 1000

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def variogram(self, distance):
        """
        The corresponding theoretical variogram to this kernel function. For
        every given distance a variogram value is estimated. The variogram
        value :math:`\gamma(d)` is estimated based on following formula with
        :math:`d` as distance and :math:`K(d)` as kernel value for given
        distance :math:`d`:

        .. math::
            \gamma(d) = K(0) - K(d)

        Parameters
        ----------
        distance : :py:class:`numpy.ndarray`
            The variogram is estimated for these distance values.

        Returns
        -------
        variogram : :py:class:`numpy.ndarray`
            The theoretical variogram values for this kernel based on given
            distance. This variogram has the same shape as given distance.
        """
        variogram = self.diag(np.zeros_like(distance))
        variogram -= self.diag(distance)
        return variogram

    def diag(self, *args, **kwargs):
        """
        The diagonal elements of this kernel.

        Parameters
        ----------
        args : list of any
            Additional arguments, which are used to estimate the diagonal
            elements.
        kwargs : dict(str, any)
            Additional keyword arguments. If no additional argument is given,
            this dictionary is used to determine the diagonal elements.

        Returns
        -------
        diag_values : any
            The diagonal values of this kernel.
        """
        return self.__call__(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _ = kwargs.pop('out', None)
        if method != '__call__':
            raise NotImplementedError
        partial_filled_ufunc = partial(ufunc, **kwargs)
        inputs = tuple(
            x if isinstance(x, BaseKernel) else Parameter(x, constant=True)
            for x in inputs
        )
        kernel_instance = KernelNode(*inputs)
        kernel_instance._np_func = partial_filled_ufunc
        return kernel_instance

    def transpose(self):
        """
        Returns a view of this kernel with axes transposed.

        Returns
        -------
        kernel : :py:class:`pylawr.remap.kernel.base_ops.UnaryKernel`
            The transposed kernel.
        """
        kernel = KernelNode(self)
        kernel._np_func = np.transpose
        return kernel

    @property
    def T(self):
        """
        Returns a view of this kernel with axes transposed.

        Returns
        -------
        kernel : :py:class:`pylawr.remap.kernel.base_ops.UnaryKernel`
            The transposed kernel.
        """
        return self.transpose()

    @property
    def params(self):
        """
        The parameters of this kernel. Parameters are optimisable during
        training and are used during inference to obtain optimized values.

        Returns
        -------
        params : list
            This are the tunable parameters of this kernel. Here this is an
            empty list.
        """
        return []

    @property
    def placeholders(self):
        """
        The placeholders of this kernel. Placeholders can be used to feed-in
        values into the kernel function.

        Returns
        -------
        placeholders : list
            This are the placeholders of this kernel. Here this is an empty
            list.
        """
        return []

    def get_named_param(self, name):
        """
        This method can be used to get a list with found parameters, where the
        name equals the given name.

        Parameters
        ----------
        name : str
            This name is searched in the list of parameter names.

        Returns
        -------
        params : list
            These are the found parameter, where the name matches the given
            name.
        """
        if not isinstance(name, str):
            raise TypeError('Given parameter name to search has to be a string')
        params = [p for p in self.params if p.name == name]
        return params


class Placeholder(BaseKernel):
    """
    This placeholder can be used to dynamically feed-in data into a defined
    kernel function. This placeholder returns one of the arguments if called.

    Parameters
    ----------
    name : str
        Name of this placeholder. This name is used to identify this
        placeholder. If this placeholder is called with additional keyword
        arguments, then this name is used to get an additional argument.
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        """
        Returns one of the additional arguments. The first additional argument
        is returned. If only keyword arguments are given, then the keyword
        argument with the name of this placeholder is returned.

        Parameters
        ----------
        args : list of any
            Additional arguments. The first additional argument is returned.
        kwargs : dict(str, any)
            Additional keyword arguments. If no additional argument is given,
            this dictionary is used to determine the return values. The name of
            this placeholder is used as identifier for this dictionary.

        Returns
        -------
        feeded_value : any
            The identified argument for this placeholder.

        Raises
        ------
        ValueError :
            A ValueError is raised if no input is given for this placeholder.
        """
        if args:
            return args[0]
        elif self.name in kwargs.keys():
            return kwargs[self.name]
        else:
            err_msg = 'No valid input for `{0:s}` is given!'.format(self.name)
            raise ValueError(err_msg)

    def __repr__(self):
        return 'placeholder("{0:s}")'.format(self.name)

    @property
    def placeholders(self):
        """
        The placeholders of this kernel. Placeholders can be used to feed-in
        values into the kernel function.

        Returns
        -------
        placeholders : list(
        :py:class:`~pylawr.remap.kernel.base_ops.Placeholder`)
            This are the placeholders of this kernel. This placeholder only
            includes an instance of this placeholder class.
        """
        return [self, ]


class Parameter(BaseKernel):
    """
    This parameter can be used to introduce variable and constant values into
    a defined kernel function.

    Parameters
    ----------
    value : any
        The value of this parameter, which is returned by calling this instance.
    name : str or None, optional
        You can name a parameter. This parameter can be then extracted by using
        :py:meth:`~pylawr.remap.kernel.base_ops.BaseKernel.get_named_param`. If
        the name of the parameter is None (default) it cannot be found.
    constant : bool, optional
        This indicates if the value is included in the parameters list (False)
        or not (True). If it is included in the parameters list, it can be
        changed during training of this kernel. In its default value, this
        parameter is trainable. Default is False.
    """
    def __init__(self, value, name=None, constant=False):
        self._name = None
        self.value = value
        self.name = name
        self.constant = constant

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str) and new_name is not None:
            raise TypeError('Given parameter name is not a string or None')
        self._name = new_name

    def __repr__(self):
        repr_value = str(self.value)
        if self.constant:
            return 'constant({0})'.format(repr_value)
        else:
            return 'parameter({0})'.format(repr_value)

    def __call__(self, *args, **kwargs):
        """
        This call returns the set value of this parameter.

        Parameters
        ----------
        args : list of any
            Additional arguments, which are not used within this call.
        kwargs : dict(str, any)
            Additional keyword arguments, which are not used within this call.

        Returns
        -------
        parameter : any
            The set value of this parameter, returned independently from given
            arguments.
        """
        return self.value

    @property
    def params(self):
        """
        A list of trainable parameter of this kernel.

        Returns
        -------
        params : list(:py:class:`~pylawr.remap.kernel.base_ops.Parameter`)
            This is the list of trainable parameters. If constant is set to
            False, this is a list with this parameter as item. If constant is
            True, then this returned parameter list is empty.
        """
        if not self.constant:
            return [self, ]
        else:
            return []


class KernelNode(BaseKernel):
    """
    A kernel node is a dependent node within a kernel definition. This node
    depends on other kernel nodes, e.g. placeholders or parameters. A `numpy
    ufunc` is used to manipulate and combine parent nodes.

    Parameters
    ----------
    *dependencies :
        iterable(child of :py:class:`~pylawr.remap.kernel.base_ops.BaseKernel`)
            This variable-length argument iterable can be used to define parent
            nodes of this dependent node.

    Warnings
    --------
    `self._np_func` has to be set before this kernel node can be used.
    """
    def __init__(self, *dependencies):
        super().__init__()
        self.dependencies = dependencies
        self._np_func = None

    def __repr__(self):
        np_repr = self._np_func.func.__name__
        dep_repr_list = [str(dep) for dep in self.dependencies]
        dep_repr = ', '.join(dep_repr_list)
        return '{0:s}({1:s})'.format(np_repr, dep_repr)

    def __call__(self, *args, **kwargs):
        dep_values = tuple(dep(*args, **kwargs) for dep in self.dependencies)
        return self.np_func(*dep_values)

    def diag(self, *args, **kwargs):
        dep_values = tuple(dep.diag(*args, **kwargs)
                           for dep in self.dependencies)
        return self.np_func(*dep_values)

    @property
    def np_func(self):
        """
        The set numpy function of this kernel. This numpy func either transforms
        or combines the parent nodes into this kernel.

        Returns
        -------
        np_func : python function
            This numpy function can be used to manipulate gained data from the
            parent nodes.

        Raises
        ------
        AttributeError
            An AttributeError is raised if not numpy function for this dependent
            kernel is set.
        """
        if self._np_func is None:
            raise AttributeError('No numpy function is set and this kernel '
                                 'cannot be evaluated!')
        return self._np_func

    @property
    def params(self):
        """
        A list of trainable parameters. These parameters can be used to train
        this kernel function onto data. This dependent kernel only forwards
        parents parameters.

        Returns
        -------
        params : list(:py:class:`~pylawr.remap.kernel.base_ops.Parameter`)
            This is the list of parameters for this kernel node. If this list is
            empty no parameter can be learned.
        """
        dep_params = (dep.params for dep in self.dependencies)
        params_chain = itertools.chain(*dep_params)
        return list(params_chain)

    @property
    def placeholders(self):
        """
        The placeholders of this kernel. Placeholders can be used to feed-in
        data into the kernel function. This dependent kernel only forwards
        parents placeholders

        Returns
        -------
        placeholders :
        list(:py:class:`~pylawr.remap.kernel.base_ops.Placeholder`)
            This is the list with available placeholders for this kernel. These
            placeholders are inherited from the dependent kernels. If this list
            is empty no placeholder can be used to feed-in data and the results
            of this kernel are static.
        """
        dep_placeholders = (dep.placeholders for dep in self.dependencies)
        placeholders_chain = itertools.chain(*dep_placeholders)
        return list(placeholders_chain)
