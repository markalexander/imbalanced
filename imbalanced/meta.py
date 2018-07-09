# -*- coding: utf-8 -, Union*-

from abc import abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Any


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

        Essentially loops through and recursively expands other objects into
        canonical dicts if available, or canonical string representations
        otherwise.

        Returns:
            The canonical dictionary representation.

        """
        cdict = OrderedDict()
        cdict['type'] = self.__class__.__name__
        cdict['args'] = OrderedDict()
        for name, value in self.args:
            cdict['args'][name] = expand(value)
        return cdict

    def __repr__(self):
        cdict = self.cdict
        return '<%s(%r)>' % (cdict['type'], cdict['args'])


def expand(obj):
    if isinstance(obj, CanonicalDictMixin):
        return obj.cdict
    elif isinstance(obj, list):
        return [expand(x) for x in obj]
    else:
        return repr(obj)
