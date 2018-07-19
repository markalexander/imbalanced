# -*- coding: utf-8 -*-

"""
This module contains utilities relating to meta- or introspection-based aspects
of the package.  Though these features may not be employed by the average user,
they are provided for convenience when running experiments, logging objects and
events, etc.

The main class here is the CanonicalArgsMixin.  This defines an interface
whereby any inheriting class should define a canonical (ordered) list of
arguments ('c-args') that represents the current object state and parameters.
These arguments are as might be used in a constructor-like string representation
for the object, e.g.:

    <ClassName(arg1=val1, arg2=val2, ...)>.

where the canonical args for this example would be the list of tuples

    [(arg1, val1), (arg2, val2), ...]

This provision is useful for:

    - Automatically building more general canonical representations (e.g.
      strings, dicts) that can be slotted in elsewhere.
    - Logging args during experiments, especially where args might be
      generated in ad hoc or grid search fashion.

"""

from abc import abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Any, Union, Dict
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer

NoneType = type(None)


class CanonicalArgsMixin:
    """Mixin for objects that provide a canonical (ordered) list of arguments
    for themselves.
    """

    @property
    @abstractmethod
    def c_args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments ('c-args') which define
        the current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        pass

    @property
    def c_dict(self) -> OrderedDict:
        """Get the canonical nested dictionary representation of the current
        object.

        Returns:
            The canonical dictionary representation.

        """
        return expand_repr(self)

    def __repr__(self) -> str:
        c_dict = self.c_dict
        return '<%s(%r)>' % (c_dict['type'], c_dict['args'])


def expand_repr(obj) -> Union[OrderedDict, List, Dict, str]:
    """Get a recursively expanded canonical-dict-based representation for a
    given object.

    N.B. this does *not* check for mutual references and infinite recursion,
    since these are very rarely used in the context.  If you need to use these,
    you should implement your own canonical representation functionality.

    Args:
        obj: The object.

    Returns:
        The expanded canonical representation.

    """

    if isinstance(obj, (int, float, str, tuple, NoneType)):
        # Basic built-ins are JSON-serializable and returned directly
        return obj

    elif isinstance(obj, list):
        # List can be expanded item-by-item
        return [expand_repr(v) for v in obj]

    elif isinstance(obj, (OrderedDict, dict)):
        # As can dicts
        return OrderedDict([(k, expand_repr(v)) for k, v in obj.items()])

    else:
        # Some kind of more complex object
        # Attempt to convert it to a dict representation

        # First, get some kind of canonical argument list
        if hasattr(obj, 'c_args'):
            # Already defined by the object if implements our mixin
            c_args = obj.c_args

        elif isinstance(obj, Optimizer):
            # Object is a torch optimizer
            grouped_params = []
            for i, group in enumerate(obj.param_groups):
                grouped_params.append([])
                for key in sorted(group.keys()):
                    if key != 'params':
                        grouped_params[i].append((key, group[key]))
            n_param_groups = len(grouped_params)
            if n_param_groups > 1:
                # Treat groups of params as enumerated args
                grouped_params = [(i, param_list)
                                  for i, param_list
                                  in enumerate(grouped_params)]
            elif n_param_groups == 1:
                # Break off the extraneous list wrapper
                grouped_params = grouped_params[0]
            c_args = grouped_params

        elif isinstance(obj, Module):
            # Object is a torch module
            c_args = []
            # Some PyTorch modules have an extra representation string with
            # e.g. layer params
            extra_repr = obj.extra_repr()
            extra_repr_args = []
            if len(extra_repr) > 0:
                extra_repr = extra_repr.split(', ')
                for argval in extra_repr:
                    extra_repr_args.append(tuple(argval.split('=')))
            # Submodules
            submodules = []
            for i, module in obj._modules.items():
                submodules.append(expand_repr(module))
            # Combine
            if len(submodules) > 0 or len(extra_repr) > 0:
                if len(submodules) > 0:
                    c_args.append(('submodules', submodules))
                if len(extra_repr_args) > 0:
                    for name, value in extra_repr_args:
                        c_args.append((name, value))

        else:
            # Object is something else, just repr() it
            c_args = [('all', repr(obj))]

        # Use the args to build and return the dict
        c_dict = OrderedDict()
        c_dict['type'] = type(obj).__name__
        if len(c_args) > 0:
            c_dict['args'] = OrderedDict()
            for arg_name, arg_val in c_args:
                c_dict['args'][arg_name] = expand_repr(arg_val)
        return c_dict
