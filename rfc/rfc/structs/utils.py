from typing import Type, TypeVar

from ..types import *
from collections.abc import Iterable

U = TypeVar('U', bound=IdentifiedObject)

def idenumerate(it: Iterable[U]):
    for x in it:
        yield x.id, x