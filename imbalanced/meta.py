# -*- coding: utf-8 -, Union*-

from abc import abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Any, Union
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer


class CanonicalDictMixin:
    """Mixin for objects that provide a canonical (ordered) dictionary
    representation of themselves.
    """

    @property
    @abstractmethod
    def args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments which define the
        current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        pass

    @property
    def cdict(self) -> OrderedDict:
        """Get the canonical dictionary representation of the current object.

        Returns:
            The canonical dictionary representation.

        """
        cdict = OrderedDict()
        cdict['type'] = self.__class__.__name__
        cdict['args'] = OrderedDict()
        for name, value in self.args:
            cdict['args'][name] = expand_repr(value)
        return cdict

    def __repr__(self):
        cdict = self.cdict
        return '<%s(%r)>' % (cdict['type'], cdict['args'])


def expand_repr(obj) -> Union[OrderedDict, List, str]:
    """Recursively expand the canonical representations of an object.

    Essentially loops through and recursively expands other objects into
    canonical dicts if available, or canonical string representations
    otherwise.

    With some specialized expansions for PyTorch objects.

    Args:
        obj: The object to be expanded.

    Returns:
        The expanded canonical representation of the given object.

    """
    if isinstance(obj, CanonicalDictMixin):
        return obj.cdict

    elif isinstance(obj, list):
        return [expand_repr(x) for x in obj]

    elif isinstance(obj, (Module, Optimizer,)):

        # Various PyTorch objects

        d = OrderedDict()
        d['type'] = obj.__class__.__name__

        if isinstance(obj, Optimizer,):
            # PyTorch optimizer
            grouped_params = []
            for i, group in enumerate(obj.param_groups):
                grouped_params.append(OrderedDict())
                for key in sorted(group.keys()):
                    if key != 'params':
                        grouped_params[i][key] = group[key]
            if len(grouped_params) == 1:
                grouped_params = grouped_params[0]
            d['args'] = grouped_params

        elif isinstance(obj, Module):
            # PyTorch module

            # Extra repr
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

            # Add them
            if len(submodules) > 0 or len(extra_repr) > 0:
                d['args'] = OrderedDict()
                if len(submodules) > 0:
                    d['args']['submodules'] = submodules
                if len(extra_repr_args) > 0:
                    for name, value in extra_repr_args:
                        d['args'][name] = value

        return d

    else:
        return repr(obj)
