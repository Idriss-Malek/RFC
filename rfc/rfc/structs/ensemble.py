import numpy as np

from collections.abc import Iterable, Iterator, Callable
from enum import Enum

from ..types import *
from .utils import *
from .feature import Feature
from .tree import Tree

def treeLevels(T: Tree, f: int | Feature) -> set[float]:
    levels = set()
    for node in T.nodes_with_feature(f):
        levels.add(node.threshold)
    return levels

def updateLevels(
    features: Iterable[Feature],
    trees: Iterable[Tree],
) -> None:
    for f in features:
        levels = set()
        for T in trees:
            levels = levels | treeLevels(T, f)
        f.levels = levels

class TreeEnsembleType(Enum):
    RF = "RF"

class Ensemble(Iterable[Tree]):
    features: tuple[Feature, ...]
    __trees: tuple[Tree, ...]
    type: TreeEnsembleType

    def __init__(
        self,
        features: Iterable[Feature],
        trees: Iterable[Tree],
        n_classes: int = 2,
        type_: str | TreeEnsembleType = TreeEnsembleType.RF,
        weigths: None | np.ndarray = None,
    ) -> None:
        self.n_classes = n_classes
        self.features = tuple(features)
        self.__trees = tuple(trees)
        if weigths is None:
            self.weights = np.ones(len(self.__trees))

        if isinstance(type_, str):
            self.type = TreeEnsembleType(type_)
        elif isinstance(type_, TreeEnsembleType):
            self.type = type_
        else:
            raise TypeError(f"type_ must be str or TreeEnsembleType, not {type(type_)}")
        updateLevels(self.numerical_features, self.__trees)

    def w(self, u: None | list[float] | dict[int, float] = None):
        w = self.weights
        if u is not None:
            for t, _ in idenumerate(self):
                try:
                    w[t] = w[t] * u[t]
                except KeyError:
                    pass
        return w

    def F(self, x: Sample):
        F = np.empty((self.n_classes, len(self)))
        for c in range(self.n_classes):
            for t, T in idenumerate(self):
                F[c, t] = T.F(x, c)
        return F

    def p(self, x: Sample, u: None | list[float] | dict[int, float] = None):
        w = self.w(u)
        return self.F(x).dot(w)

    def klass(
        self,
        x: Sample,
        u: None | list[float] = None,
        tiebreaker: None | Callable[[Iterable[int]], int] = None
    ) -> int:
        p = self.p(x, u)
        if tiebreaker is None:
            return int(np.argmax(p))
        else:
            a = np.argwhere(p == np.amax(p))
            return tiebreaker(a)

    @property
    def numerical_features(self) -> tuple[Feature, ...]:
        return tuple(filter(lambda f: f.isnumerical(), self.features))

    @property
    def binary_features(self) -> tuple[Feature, ...]:
        return tuple(filter(lambda f: f.isbinary(), self.features))

    def __getitem__(self, idx: int) -> Tree:
        return self.__trees[idx]

    def __iter__(self) -> Iterator[Tree]:
        return self.__trees.__iter__()

    def __len__(self) -> int:
        return len(self.__trees)

    def __repr__(self) -> str:
        return f"TreeEnsemble: type:{self.type}, n_trees={len(self.__trees)}\n\n{''.join(map(repr, self.__trees))}"