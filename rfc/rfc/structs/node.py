from typing import Any
from anytree import NodeMixin
from anytree import TreeError
from collections.abc import Iterable

from ..types import *
from .feature import Feature, FeatureType

class BaseNode(object):
    pass

class Node(IdentifiedObject, BaseNode, NodeMixin):
    name: str
    __feature: Feature
    __threshold: float
    __categories: tuple
    __klass: int

    """
    Node of a decision tree. A node can be either a leaf or an internal node.
    
    An internal node has a feature, two children. If the feature is numerical,
    a threshold is also defined. If the feature is categorical, a list of
    categories is defined.
    
    A leaf node has a class.
    """
    def __init__(
        self,
        id_: int,
        name: None | str = None,
        feature: None | Feature = None,
        threshold: None | float = None,
        categories: None | Iterable[Any] = None,
        klass: None | int = None,
        parent = None,
        left = None,
        right = None,
    ) -> None:
        super(Node, self).__init__()
        self.id = id_
        self.name = name if name is not None else f'Node {id_}'
        self.parent = parent
        if left: self.left = left
        if right: self.right = right
        if feature: self.__feature = feature
        if threshold: self.__threshold = threshold
        if categories: self.__categories = tuple(categories)
        if klass: self.__klass = klass

    @property
    def feature(self) -> Feature:
        if self.is_leaf:
            msg = "Cannot get the feature of a leaf node!"
            raise AttributeError(msg)
        try:
            return self.__feature
        except AttributeError:
            msg = "The feature of this node is not defined!"
            raise AttributeError(msg)

    @feature.setter
    def feature(self, feature: Feature):
        if self.is_leaf:
            msg = "Cannot set a feature for a leaf node!"
            raise AttributeError(msg)

        try:
            self.__feature
            msg = "The feature of this node is already defined!"
            raise AttributeError(msg)
        except AttributeError:
            self.__feature = feature

    @property
    def threshold(self) -> float:
        if self.is_leaf:
            msg = "Cannot get the threshold of a leaf node!"
            raise AttributeError(msg)

        if not self.feature.isnumerical():
            msg = "Cannot get the threshold of a non-numerical feature node!"
            raise AttributeError(msg)

        try:
            return self.__threshold
        except AttributeError:
            msg = "The threshold of this node is not defined!"
            raise AttributeError(msg)

    @threshold.setter
    def threshold(self, threshold: float):
        if self.is_leaf:
            msg = "Cannot set a threshold for a leaf node!"
            raise AttributeError(msg)

        if not self.feature.isnumerical():
            msg = "Cannot set a threshold for a non-numerical feature node!"
            raise AttributeError(msg)

        self.__threshold = threshold

    @property
    def categories(self) -> tuple[Any]:
        if self.is_leaf:
            msg = "Cannot get the categories of a leaf node!"
            raise AttributeError(msg)

        if not self.feature.iscategorical():
            msg = "Cannot get the categories of a non-categorical feature node!"
            raise AttributeError(msg)

        try:
            return self.__categories
        except AttributeError:
            msg = "The categories of this node are not defined!"
            raise AttributeError(msg)

    @categories.setter
    def categories(self, categories: Iterable[Any]):
        if self.is_leaf:
            msg = "Cannot set categories for a leaf node!"
            raise AttributeError(msg)

        if not self.feature.iscategorical():
            msg = "Cannot set categories for a non-categorical feature node!"
            raise AttributeError(msg)
        
        self.__categories = tuple(categories)

    @property
    def klass(self) -> int:
        if not self.is_leaf:
            msg = "Cannot get the class of a non-leaf node!"
            raise AttributeError(msg)

        try:
            return self.__klass
        except AttributeError:
            msg = "The class of this leaf is not defined!"
            raise AttributeError(msg)

    @klass.setter
    def klass(self, klass: int):
        if not self.is_leaf:
            msg = "Cannot set a class for a non-leaf node!"
            raise AttributeError(msg)
        self.__klass = klass

    @property
    def left(self) -> "Node":
        if self.is_leaf:
            msg = "Cannot get the left child of a leaf node!"
            raise TreeError(msg)
        return self.__left

    @left.setter
    def left(self, node):
        try:
            left = self.__left
        except AttributeError:
            left = None
        if left is not node:
            self.__detach_child(left)
            self.__attach_child(node)
            self.__left = node

    @left.deleter
    def left(self):
        try:
            left = self.__left
        except AttributeError:
            left = None
        self.__detach_child(left)

    @property
    def right(self) -> "Node":
        if self.is_leaf:
            msg = "Cannot get the right child of a leaf node!"
            raise TreeError(msg)
        return self.__right

    @right.setter
    def right(self, node):
        try:
            right = self.__right
        except AttributeError:
            right = None
        if right is not node:
            self.__detach_child(right)
            self.__attach_child(node)
            self.__right = node

    def p(self, c: int) -> float:
        return 0.0 + (self.klass == c)

    def __f(self, value) -> bool:
        match self.feature.type:
            case FeatureType.CATEGORICAL:
                return value in self.categories
            case FeatureType.NUMERICAL:
                return value <= self.threshold
            case FeatureType.BINARY:
                return value == 0

    def split_on(self, f: int | Feature) -> bool:
        return self.feature.id == f if isinstance(f, int) else self.feature == f

    def split(self, x: Sample):
        v = self.feature.value(x)
        return self.__split(v)

    def __split(self, value: float | int | Any):
        if self.is_leaf:
            msg = "Cannot split a leaf node!"
            raise ValueError(msg)
        return self.left if self.__f(value) else self.right

    def __detach_child(self, child):
        if child is not None:
            self.children = (node for node in self.children if node is not child)
            child.parent = None

    def __attach_child(self, child):
        if child is not None: child.parent = self

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"<Leaf id={self.id}, name={self.name}, class={self.klass}>"
        elif self.feature.isnumerical():
            return f"<Node id={self.id}, name={self.name}, feature={self.feature}, thr={self.threshold}>"
        elif self.feature.iscategorical():
            return f"<Node id={self.id}, name={self.name}, feature={self.feature}, cats={self.categories}>"
        else:
            return f"<Node id={self.id}, name={self.name}, feature={self.feature}>"