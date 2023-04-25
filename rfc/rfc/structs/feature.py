from typing import Any
from enum import Enum

class FeatureType(Enum):
    NUMERICAL = 0
    CATEGORICAL = 1
    BINARY = 2

class Feature(object):
    name: str
    id: int
    ftype: FeatureType
    _levels: None | list[float] = None
    _categories: None | list[Any] = None

    def __init__(
        self,
        id_: int,
        ftype: FeatureType,
        name: None | str = None
    ) -> None:
        self.id = id_
        self.name = name if name is not None else f'Feature {id_}'
        self.ftype = ftype

    @property
    def levels(self) -> list[float]:
        if self.ftype != FeatureType.NUMERICAL:
            raise ValueError("Levels are only defined for numerical features!")
        if self._levels is None:
            raise ValueError("Feature has no levels!")
        return self._levels

    @levels.setter
    def levels(self, levels: list[float]):
        if self.ftype != FeatureType.NUMERICAL:
            raise ValueError("Levels are only defined for numerical features!")
        self._levels = levels

    @property
    def categories(self) -> list[Any]:
        if self.ftype != FeatureType.CATEGORICAL:
            raise ValueError("Values are only defined for categorical features!")
        if self._categories is None:
            raise ValueError("Feature has no values!")
        return self._categories

    @categories.setter
    def categories(self, categories: list[Any]):
        if self.ftype != FeatureType.CATEGORICAL:
            raise ValueError("Values are only defined for categorical features!")
        self._categories = categories

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.id}, {self.ftype})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Feature):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)