# -*- coding: utf-8 -*-

from abc import abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Any


class CanonicalDictMixin:
    """Mixin for objects that provide a canonical (ordered) dictionary
    representation of themselves.
    """

    @property
    @abstractmethod
    def cdict(self) -> OrderedDict:
        """Get the canonical dict representation of the current object.

        Returns:
            The canonical dict representation.

        """
        pass

    def _cdict_from_args(self, args: List[Tuple[str, Any]]) -> OrderedDict:
        """Construct a canonical (ordered) dictionary from a list of arg tuples.

        Essentially loops through and recursively expands other objects into
        canonical dicts if available, or canonical string representations
        otherwise.

        Args:
            args: A list of arguments as in the constructor of the object, with
                  elements of the form (arg_name, arg_value).

        Returns:

        """
        r_args = []
        for name, value in args:
            if isinstance(value, self.__class__):
                r_args.append((name, value.cdict))
            else:
                r_args.append((name, repr(value)))
        return OrderedDict(r_args)

    def __repr__(self):
        args = ', '.join(['%s=%s' % item for item in self.cdict.items()])
        return '<%s(%s)>' % (self.__class__.__name__, args)
