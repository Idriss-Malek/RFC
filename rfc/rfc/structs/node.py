from typing import Any
from anytree import NodeMixin

from .feature import Feature, FeatureType

class BaseNode(object):
    pass

class Node(BaseNode, NodeMixin):
    id: int
    name: str
    _feature: None | Feature = None
    _threshold: None | float = None
    _categories: None | list[Any] = None
    _leftIdx: None | int = None
    _rightIdx: None | int = None
    _klass: None | int = None

    def __init__(
        self,
        id_: int,
        name: None | str = None,
        feature: None | Feature = None,
        threshold: None | float = None,
        values: None | list[Any] = None,
        klass: None | int = None,
        parent = None,
        children: None | tuple = None,
    ) -> None:
        super(Node, self).__init__()
        self.id = id_
        self.name = name if name is not None else f'Node {id_}'
        self.parent = parent
        if children:  # Set children only if they are not None
            if len(children) > 2:
                raise ValueError("Binary node can only have two children.")
            if len(children) >= 1:
                self._leftIdx = 0
            if len(children) == 2:
                self._rightIdx = 1
            self.children = children
        if feature is not None:
            self.feature = feature
        if threshold is not None:
            self.threshold = threshold
        if values is not None:
            self.categories = values
        if klass is not None:
            self.klass = klass

    @property
    def feature(self) -> Feature:
        if self.is_leaf:
            raise ValueError("Leaf node has no feature!")
        if self._feature is None:
            raise ValueError("Node has no feature!")
        return self._feature

    @feature.setter
    def feature(self, feature: Feature):
        if self.is_leaf:
            raise ValueError("Leaf node has no feature!")
        self._feature = feature

    @property
    def threshold(self) -> float:
        if self.is_leaf:
            raise ValueError("Leaf node has no threshold!")
        if self.feature.ftype != FeatureType.NUMERICAL:
            raise ValueError("Threshold is only defined for numerical features!")
        if self._threshold is None:
            raise ValueError("Node has no threshold!")
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float):
        if self.is_leaf:
            raise ValueError("Leaf node has no threshold!")
        if self.feature.ftype != FeatureType.NUMERICAL:
            raise ValueError("Threshold is only defined for numerical features!")
        self._threshold = threshold

    @property
    def categories(self) -> list[Any]:
        if self.is_leaf:
            raise ValueError("Leaf node has no categories!")
        if self.feature.ftype != FeatureType.CATEGORICAL:
            raise ValueError("categories are only defined for categorical features!")
        if self._categories is None:
            raise ValueError("Node has no categories!")
        return self._categories

    @categories.setter
    def categories(self, values: list[Any]):
        if self.is_leaf:
            raise ValueError("Leaf node has no categories!")
        if self.feature.ftype != FeatureType.CATEGORICAL:
            raise ValueError("Categories are only defined for categorical features!")
        self._categories = values

    @property
    def klass(self) -> int:
        if not self.is_leaf:
            raise ValueError("Node is not a leaf!")
        if self._klass is None:
            raise ValueError("Leaf node has no class!")
        return self._klass

    @klass.setter
    def klass(self, klass: int):
        if not self.is_leaf:
            raise ValueError("Node is not a leaf!")
        self._klass = klass

    @property
    def left(self) -> "Node":
        if self._leftIdx is None:
            raise ValueError("No left child!")
        return self.children[self._leftIdx]

    @left.setter
    def left(self, node: "Node"):
        if self._leftIdx is not None:
            raise ValueError("Node has already a left child!")
        if len(self.children) > 2:
            raise ValueError("Node has already two children!")
        self._leftIdx = len(self.children)
        node.parent = self

    @property
    def right(self) -> "Node":
        if self._rightIdx is None:
            raise ValueError("No right child!")
        return self.children[self._rightIdx]

    @right.setter
    def right(self, node: "Node"):
        if self._rightIdx is not None:
            raise ValueError("Node has already a right child!")
        if len(self.children) > 2:
            raise ValueError("Node has already two children!")
        self._rightIdx = len(self.children)
        node.parent = self

    def p(self, c: int) -> float:
        return (self.klass == c)+0.
        
    def next(self, value: float | int | Any) -> "Node":
        if self.is_leaf:
            raise ValueError("Leaf node has no next.")

        match self.feature.ftype:
            case FeatureType.BINARY:
                return self.left if value == 0 else self.right
            case FeatureType.CATEGORICAL:
                return self.left if (value in self.categories) else self.right
            case FeatureType.NUMERICAL:
                return self.left if value <= self.threshold else self.right