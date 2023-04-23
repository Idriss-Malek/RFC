import numpy as np
import pandas as pd

from dataclasses import dataclass
from anytree import NodeMixin, PreOrderIter
from collections.abc import Iterable, Iterator, Callable

import pathlib

@dataclass
class BaseTreeNode(object):
    feature: int
    thr: float
    klass: int

class TreeNode(BaseTreeNode, NodeMixin):
    name: str
    id: int

    def __init__(
        self,
        name,
        id_: int,
        feature: int,
        thr: float,
        klass: int,
        parent = None,
        children: None | tuple = None,
    ) -> None:
        super(TreeNode, self).__init__(feature, thr, klass)
        self.name = name
        self.id = id_
        self.parent = parent
        if children:  # Set children only if they are not None
            self.children = children

    @property
    def left(self) -> "TreeNode":
        if self.is_leaf or len(self.children) == 0:
            raise ValueError("No left child")
        return self.children[0]

    @left.setter
    def left(self, node: "TreeNode"):
        if not self.is_leaf and len(self.children) >= 1:
            raise ValueError("Node has already a left child")
        node.parent = self

    @property
    def right(self) -> "TreeNode":
        if self.is_leaf or len(self.children) < 2:
            raise ValueError("No right child")
        return self.children[1]

    @right.setter
    def right(self, node: "TreeNode"):
        if not self.is_leaf and len(self.children) >= 2:
            raise ValueError("Node has already a right child")
        if len(self.children) == 0:
            raise ValueError("Node has no left child")
        node.parent = self

    def getNNodes(self) -> int:
        return len(self.descendants) + 1

class Tree(Iterable[TreeNode]):
    name: str
    root: TreeNode
    n_nodes: int

    def __init__(
        self,
        name: str,
        root: TreeNode
    ) -> None:
        self.name = name
        self.root = root
        self.n_nodes = root.getNNodes()

    @classmethod
    def from_lines(cls, lines: list[str], start: int = 0):
        assert "[TREE" in lines[0]
        root = None
        n_nodes = int(lines[start + 1].split(": ")[1])
        parents: dict[int, TreeNode] = {}
        n = 0
        while n < n_nodes:
            line = lines[n + start + 2]
            idx, ntype, left, right, feature, thr, _, klass = line.split()
            name = idx
            idx = int(idx)
            feature = int(feature)
            thr = float(thr)
            klass = int(klass)
            left = int(left)
            right = int(right)
            node = TreeNode(name=name, id_=idx, feature=feature, thr=thr, klass=klass)
            if node.id in parents:
                node.parent = parents[node.id]
            else:
                root = node
            if ntype == "IN":
                parents[left] = node
                parents[right] = node
            n += 1
        name = "tree"
        assert(root is not None and isinstance(root, TreeNode))
        return cls(name, root)  # type: ignore

    def getKlass(self, x: np.ndarray):
        node = self.root
        while not node.is_leaf:
            if x[node.feature] <= node.thr:
                node = node.left
            else:
                node = node.right
        return node.klass

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
        return np.array([leaf.klass == c for leaf in self.getLeaves()])

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
        for node in PreOrderIter(self.root):
            yield node

    def __len__(self) -> int:
        return self.n_nodes

    @property
    def m_depth(self) -> int:
        return self.root.height

class TreeEnsemble(Iterable[Tree]):
    def __init__(
        self,
        dataset: str = "train.csv",
        etype: str = "RF",
        n_classes: int = 0,
        n_features: int = 0,
        m_depth: int = 0,
        trees: None | list[Tree] = None,
        weigths: None | np.ndarray = None,
        numerical_features: None | Iterable[int] = None,
        binary_features: None | Iterable[int] = None,
        categorical_features: None | Iterable[int] = None,
    ) -> None:
        self.dataset = dataset
        self.etype = etype
        self.n_classes = n_classes
        self.n_features = n_features
        self.m_depth = m_depth
        if trees is None:
            self.trees = []
        else:
            self.trees = trees
        if weigths is None:
            self.weights = np.ones(len(self.trees))

        self._updateFeatures(numerical_features, binary_features, categorical_features)
        self.numerical_levels = self.getNumericalLevels() # type: dict[int, list[float]]
        self.categorical_values = self.getCategoricalValues() # type: dict[int, list[int]]

    @classmethod
    def from_file(cls, file: str):
        with open(file, "r") as f:
            lines = f.readlines()
            f.close()
        dataset = lines[0].split(": ")[1][:-1]
        etype = lines[1].split(": ")[1]
        n_trees = int(lines[2].split(": ")[1])
        n_features = int(lines[3].split(": ")[1])
        n_classes = int(lines[4].split(": ")[1])
        m_depth = int(lines[5].split(": ")[1])
        t = 0
        lineIdx = 8
        trees = []
        while t < n_trees:
            tree = Tree.from_lines(lines[lineIdx:])
            trees.append(tree)
            lineIdx += tree.n_nodes + 3
            t += 1
        return cls(
            dataset=dataset,
            etype=etype,
            n_classes=n_classes,
            n_features=n_features,
            m_depth=m_depth,
            trees=trees,
        )

    def getF(self, x: np.ndarray):
        F = np.empty((self.n_classes, self.getNTrees()))
        for c in range(self.n_classes):
            for t, tree in enumerate(self):
                F[c, t] = tree.getF(x, c)
        return F

    def getNumericalLevels(self) -> dict[int, list[float]]:
        res: dict[int, list[float]] = {f: [] for f in self.numerical_features}
        
        for f in self.numerical_features:
            for tree in self:
                for node in tree:
                    if node.feature == f:
                        res[f].append(node.thr)

        for f in self.numerical_features:
            levels = np.array(list(res[f]))
            levels = np.sort(levels)
            levels = 0.5 * (1 + np.tanh(levels))
            res[f] = list(levels)
            res[f] = [0.0] + res[f] + [1.0]
            
        return {f: levels for f, levels in res.items()}
    
    def getCategoricalValues(self) -> dict[int, list[int]]:
        return {}

    def getNTrees(self):
        return len(self)

    def _updateFeatures(
        self,
        numerical_features: None | Iterable[int],
        binary_features: None | Iterable[int],
        categorical_features: None | Iterable[int]
    ):
        all_features = set(range(self.n_features))
        if numerical_features is not None:
            self.numerical_features = set(numerical_features)
        if binary_features is not None:
            self.binary_features = set(binary_features)
        if categorical_features is not None:
            self.categorical_features = set(categorical_features)

        cnt = len(list(filter(lambda x: x is not None, [numerical_features, binary_features, categorical_features])))
        match cnt:
            case 0:
                self.numerical_features = all_features
                self.binary_features = set()
                self.categorical_features = set()
            case 1:
                if numerical_features is not None:
                    self.binary_features = all_features - self.numerical_features
                    self.categorical_features = set()
                elif binary_features is not None:
                    self.numerical_features = all_features - self.binary_features
                    self.categorical_features = set()
                elif categorical_features is not None:
                    self.numerical_features = all_features - self.categorical_features
                    self.binary_features = set()
            case 2:
                if numerical_features is None:
                    self.numerical_features = all_features - (self.binary_features | self.categorical_features)
                elif binary_features is None:
                    self.binary_features = all_features - (self.numerical_features | self.categorical_features)
                elif categorical_features is None:
                    self.categorical_features = all_features - (self.numerical_features | self.binary_features)
            case _:
                pass
        assert(len(self.numerical_features | self.binary_features | self.categorical_features) == self.n_features)

    def __getitem__(self, idx: int) -> Tree:
        return self.trees[idx]

    def __iter__(self) -> Iterator[Tree]:
        return self.trees.__iter__()

    def __len__(self) -> int:
        return len(self.trees)