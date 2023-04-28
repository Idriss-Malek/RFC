from typing import Any
from collections.abc import Iterable
from enum import Enum

from ..types import *

class FeatureType(Enum):
    NUMERICAL = 0
    CATEGORICAL = 1
    BINARY = 2

class Feature(IdentifiedObject):
    name: str
    type: FeatureType
    _levels: None | list[float] = None
    _categories: None | list[Any] = None

    def __init__(
        self,
        id_: int,
        type_: FeatureType,
        name: None | str = None,
        levels: None | Iterable[float] = None,
        categories: None | list[Any] = None,
    ) -> None:
        self.id = id_
        self.name = name if name is not None else f'Feature {id_}'
        self.type = type_
        if levels is not None:
            self.levels = levels
        if categories is not None:
            self.categories = categories

    def isnumerical(self) -> bool:
        return self.type == FeatureType.NUMERICAL

    def isbinary(self) -> bool:
        return self.type == FeatureType.BINARY

    def iscategorical(self) -> bool:
        return self.type == FeatureType.CATEGORICAL

    def value(self, x: Sample) -> Any:
        return x[self.id]

    @property
    def levels(self) -> list[float]:
        if not self.isnumerical():
            raise ValueError("Levels are only defined for numerical features!")
        if self._levels is None:
            raise ValueError("Feature has no levels!")
        return self._levels

    @levels.setter
    def levels(self, levels: Iterable[float]):
        if not self.isnumerical():
            raise ValueError("Levels are only defined for numerical features!")
        self._levels = list(sorted(levels))
                
    @property
    def categories(self) -> list[Any]:
        if not self.iscategorical():
            raise ValueError("Categories are only defined for categorical features!")
        if self._categories is None:
            raise ValueError("Feature has no categories!")
        return self._categories

    @categories.setter
    def categories(self, categories: list[Any]):
        if not self.iscategorical():
            raise ValueError("Categories are only defined for categorical features!")
        self._categories = categories

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.id}, {self.type})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Feature):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)