import numpy as np

from dataclasses import dataclass
from anytree import NodeMixin, PreOrderIter
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Callable
from typing import Any

from enum import Enum

class FeatureType(Enum):
    NUMERICAL = 'F'
    CATEGORICAL = 'D'
    BINARY = 'B'

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
    def categories(self, values: list[Any]):
        if self.ftype != FeatureType.CATEGORICAL:
            raise ValueError("Values are only defined for categorical features!")
        self._categories = values

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

class NodeType(Enum):
    INTERNAL = 'IN'
    LEAF = 'LN'

class BaseTreeNode(object):
    pass

class TreeNode(BaseTreeNode, NodeMixin):
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
        super(TreeNode, self).__init__()
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
            raise ValueError("Leaf node has no values!")
        if self.feature.ftype != FeatureType.CATEGORICAL:
            raise ValueError("Values are only defined for categorical features!")
        if self._categories is None:
            raise ValueError("Node has no values!")
        return self._categories

    @categories.setter
    def categories(self, values: list[Any]):
        if self.is_leaf:
            raise ValueError("Leaf node has no values!")
        if self.feature.ftype != FeatureType.CATEGORICAL:
            raise ValueError("Values are only defined for categorical features!")
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
    def left(self) -> "TreeNode":
        if self._leftIdx is None:
            raise ValueError("No left child!")
        return self.children[self._leftIdx]

    @left.setter
    def left(self, node: "TreeNode"):
        if self._leftIdx is not None:
            raise ValueError("Node has already a left child!")
        if len(self.children) > 2:
            raise ValueError("Node has already two children!")
        self._leftIdx = len(self.children)
        node.parent = self

    @property
    def right(self) -> "TreeNode":
        if self._rightIdx is None:
            raise ValueError("No right child!")
        return self.children[self._rightIdx]

    @right.setter
    def right(self, node: "TreeNode"):
        if self._rightIdx is not None:
            raise ValueError("Node has already a right child!")
        if len(self.children) > 2:
            raise ValueError("Node has already two children!")
        self._rightIdx = len(self.children)
        node.parent = self

    def next(self, value: float | int | Any) -> "TreeNode":
        if self.is_leaf:
            raise ValueError("Leaf node has no next.")

        match self.feature.ftype:
            case FeatureType.BINARY:
                return self.left if value == 0 else self.right
            case FeatureType.CATEGORICAL:
                return self.left if (value in self.categories) else self.right
            case FeatureType.NUMERICAL:
                assert self._threshold is not None
                return self.left if value <= self.threshold else self.right

class Tree(Iterable[TreeNode]):
    id: int
    name: str
    root: TreeNode

    def __init__(
        self,
        id_: int,
        root: TreeNode,
        name: None | str = None,
    ) -> None:
        self.id = id_
        self.name = name if name is not None else f'Tree {id_}'
        self.root = root

    def getLeaf(self, x: np.ndarray) -> TreeNode:
        node = self.root
        while not node.is_leaf:
            node = node.next(x[node.feature.id])
        return node

    def getKlass(self, x: np.ndarray):
        return self.getLeaf(x).klass

    def getF(self, x: np.ndarray, c: int):
        return 1.0 if self.getKlass(x) == c else 0.0

    def getNodes(
        self,
        filter_: None | Callable[[TreeNode], bool] = None,
        leaves: bool = True
    ) -> Iterator[TreeNode]:
        def filter__(node: TreeNode) -> bool:
            return (leaves or not node.is_leaf) and (filter_ is None or filter_(node))
        for node in PreOrderIter(self.root, filter_=filter__):
            yield node

    def getProbas(self, c: int):
        return np.array([leaf.klass == c for leaf in self.getLeaves()], dtype=np.float32)

    def getLeaves(self) -> Iterator[TreeNode]:
        return self.root.leaves.__iter__()

    def getNodesAtDepth(
        self,
        depth: int,
        leaves: bool = True
    ) -> Iterator[TreeNode]:
        return self.getNodes(lambda node: node.depth == depth, leaves)

    def getNodesWithFeature(
        self,
        feature: int,
        leaves: bool = True
    ) -> Iterator[TreeNode]:
        return self.getNodes(lambda node: node.feature == feature, leaves)

    def __iter__(self) -> Iterator[TreeNode]:
        return self.getNodes()

    def __len__(self) -> int:
        return len(self.root.descendants) + 1

    @property
    def depth(self) -> int:
        return self.root.height

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

    @classmethod
    def from_file(cls, file: str):
        with open(file, "r") as f:
            lines = f.readlines()
            f.close()

        # dataset = lines[0].split(": ")[1]
        etype = lines[1].split(": ")[1]
        n_trees = int(lines[2].split(": ")[1])
        n_features = int(lines[3].split(": ")[1])
        n_classes = int(lines[4].split(": ")[1])
        # m_depth = int(lines[5].split(": ")[1])
        lineIdx = 8
        
        lineIdx += 1
        features: list[Feature] = []
        for f in range(n_features):
            line = lines[lineIdx]
            name, ftype = line.strip().split(": ")
            ftype = FeatureType(ftype)
            feature = Feature(
                id_=f,
                name=name,
                ftype=ftype
            )
            if ftype == FeatureType.CATEGORICAL:
                lineIdx += 1
                line = lines[lineIdx]
                values = list(map(int, line.split()))
                feature.categories = values
            features.append(feature)
            lineIdx += 1
        
        lineIdx += 1

        trees = []
        t = 0
        while t < n_trees:
            lineIdx += 1
            line = lines[lineIdx]
            root = None
            n_nodes = int(line.split(": ")[1])
            parents: dict[int, TreeNode] = {}
            n = 0
            lineIdx += 1
            while n < n_nodes:
                line = lines[lineIdx]
                nodeId, ntype, leftId, rightId, featureId, thr, _, klass = line.split()
                nodeId = int(nodeId)
                ntype = NodeType(ntype)
                featureId = int(featureId)
                thr = float(thr)
                klass = int(klass)
                leftId = int(leftId)
                rightId = int(rightId)
                match ntype:
                    case NodeType.LEAF:
                        node = TreeNode(id_=nodeId, klass=klass)
                    case NodeType.INTERNAL:
                        feature = features[featureId]
                        node = TreeNode(id_=nodeId, feature=feature)
                        match feature.ftype:
                            case FeatureType.CATEGORICAL:
                                node.categories = []
                            case FeatureType.NUMERICAL:
                                node.threshold = thr
                        parents[leftId] = node
                        parents[rightId] = node
                if node.id in parents:
                    node.parent = parents[node.id]
                else:
                    root = node
                lineIdx += 1
                n += 1

            assert root is not None
            tree = Tree(id_ = t, root = root)
            trees.append(tree)
            t += 1
            lineIdx += 2
    
        return cls(
            trees=trees,
            features=features,
            n_classes=n_classes,
            etype=etype
        )

    def getF(self, x: np.ndarray):
        F = np.empty((self.n_classes, self.__len__()))
        for c in range(self.n_classes):
            for t, tree in enumerate(self):
                F[c, t] = tree.getF(x, c)
        return F

    def _updateNumericalLevels(self):
        for feature in self.features:
            if feature.ftype == FeatureType.NUMERICAL:
                levels = []
                for tree in self:
                    levels += [node.threshold for node in tree.getNodesWithFeature(feature.id)]
                levels = np.array(levels)
                levels = np.sort(levels)
                levels = 0.5 * (1 + np.tanh(levels))
                feature.levels = list(levels)
                feature.levels = [0.0] + feature.levels + [1.0]

    def getCategoricalValues(self) -> dict[int, list[int]]:
        return {}

    def __getitem__(self, idx: int) -> Tree:
        return self.trees[idx]

    def __iter__(self) -> Iterator[Tree]:
        return self.trees.__iter__()

    def __len__(self) -> int:
        return len(self.trees)