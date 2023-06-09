from typing import Type, TypeVar

from ..types import *
from collections.abc import Iterable

U = TypeVar('U', bound=IdentifiedObject)

def idenumerate(it: Iterable[U]):
    for x in it:
        yield x.id, x

def argmin( it, comp = None):
    x = it[0]
    if comp is None:
        comp = lambda a,b : a<b
    for i in it:
        if comp(i,x):
            x = i
    return x