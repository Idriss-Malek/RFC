import numpy as np

from collections.abc import Iterable, Iterator

from enum import Enum

from .feature import Feature, FeatureType
from .tree import Tree

class NodeType(Enum):
    INTERNAL = 'IN'
    LEAF = 'LN'

class ChildType(Enum):
    LEFT = 'L'
    RIGHT = 'R'

class TreeEnsemble(Iterable[Tree]):
    _features: list[Feature]

    def __init__(
        self,
        features: list[Feature],
        trees: list[Tree],
        n_classes: int = 2,
        etype: str = "RF",
        weigths: None | np.ndarray = None,
    ) -> None:
        self.n_classes = n_classes
        self.features = features
        self.trees = trees
        if weigths is None:
            self.weights = np.ones(len(self.trees))

        self.etype = etype
        self._updateNumericalLevels()


    @property
    def features(self) -> list[Feature]:
        return self._features

    @features.setter
    def features(self, features: list[Feature]):
        self._features = features

    def getF(self, x: np.ndarray):
        F = np.empty((self.n_classes, self.__len__()))
        for c in range(self.n_classes):
            for t, tree in enumerate(self):
                F[c, t] = tree.getF(x, c)
        return F

    def _updateNumericalLevels(self):
        for feature in self.features:
            if feature.ftype == FeatureType.NUMERICAL:
                levels = set()
                for tree in self:
                    tree_levels = [node.threshold for node in tree.getNodesWithFeature(feature.id)]
                    levels = levels | set(tree_levels)
                feature.levels = levels

    def __getitem__(self, idx: int) -> Tree:
        return self.trees[idx]

    def __iter__(self) -> Iterator[Tree]:
        return self.trees.__iter__()

    def __len__(self) -> int:
        return len(self.trees)