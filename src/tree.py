import numpy as np
import pandas as pd

from anytree import NodeMixin, PreOrderIter

import pathlib

class BaseNode(object):
    pass


class TreeNode(BaseNode, NodeMixin):
    feature: int
    thr: float
    klass: int

    def __init__(
        self,
        name,
        feature: int,
        thr: float,
        klass: int,
        parent=None,
        children: None | tuple = None,
    ) -> None:
        super(TreeNode, self).__init__()
        self.name = name
        self.feature = feature
        self.thr = thr
        self.klass = klass
        self.parent = parent
        if children:  # Set children only if they are not None
            self.children = children


class Tree:
    name: str
    root: TreeNode
    n_nodes: int

    def __init__(self, name: str, root: TreeNode, n_nodes: int) -> None:
        self.name = name
        self.root = root
        self.n_nodes = n_nodes

    @classmethod
    def from_lines(cls, lines: list[str], start: int = 0):
        assert "[TREE" in lines[0]
        name = "tree"
        root = None
        n_nodes = int(lines[start + 1].split(": ")[1])
        parents = {}
        n = 0
        while n < n_nodes:
            line = lines[n + start + 2]
            idx, ntype, left, right, feature, thr, _, klass = line.split()
            feature = int(feature)
            thr = float(thr)
            klass = int(klass)
            node = TreeNode(name=idx, feature=feature, thr=thr, klass=klass)
            if node.name in parents:
                node.parent = parents[node.name]
            else:
                root = node
            if ntype == "IN":
                parents[left] = node
                parents[right] = node
            n += 1
        return cls(name, root, n_nodes)  # type: ignore

    def getKlass(self, x: pd.Series):
        node = self.root
        while node.is_leaf is False:
            if x[node.feature] <= node.thr:
                node = node.children[0]
            else:
                node = node.children[1]
        return node.klass

    def getF(self, x: pd.Series, c: int):
        return 1.0 if self.getKlass(x) == c else 0.0


class TreeEnsemble:
    def __init__(
        self,
        dataset: str = "train.csv",
        etype: str = "RF",
        n_classes: int = 0,
        n_features: int = 0,
        m_depth: int = 0,
        trees: None | list[Tree] = None,
        weigths: None | np.ndarray = None,
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
            self.weigths = np.ones(len(self.trees))

        self.num_levels = self.getNumLevels()  # type: dict[int, list[float]]
        self.bin = self.getBinFeatures() #type: list[int]

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

    def getF(self, x: pd.Series):
        F = np.empty((self.n_classes, len(self.trees)))
        for c in range(self.n_classes):
            for t in range(len(self.trees)):
                F[c, t] = self.trees[t].getF(x, c)
        return F

    def getNumLevels(self) -> dict[int, list[float]]:
        res = {}
        root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
        data=str(root / ('datasets/'+self.dataset.split('.')[0]+'/'+self.dataset))
        df = pd.read_csv(str(root / ('datasets/'+self.dataset.split('.')[0]+'/'+self.dataset)))
        for f in range(self.n_features):
            if len(df.iloc[:, f].unique()) >= 3:
                levels = set()
                for t, tree in enumerate(self.trees):
                    for node in PreOrderIter(tree.root):
                        if node.feature == f:
                            levels.add(node.thr)
                levels.update({0.0, 1.0})
            res[f] = levels
        return res

    def getBinFeatures(self) -> list[int]:
        res = []
        root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
        data=str(root / ('datasets/'+self.dataset.split('.')[0]+'/'+self.dataset))
        df = pd.read_csv(data)
        for f in range(self.n_features):
            if len(df.iloc[:, f].unique()) == 2:
                res.append(f)
        return res

