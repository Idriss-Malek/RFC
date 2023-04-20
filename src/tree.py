import numpy as np
import pandas as pd

from anytree import NodeMixin

class BaseNode(object):
    pass

class TreeNode(BaseNode, NodeMixin):
    def __init__(
        self,
        name,
        feature: int,
        thr: float,
        klass: int,
        parent = None,
        children: None | tuple = None
    ) -> None:
        super(TreeNode, self).__init__()
        self.name = name
        self.feature = feature
        self.thr = thr
        self.klass = klass
        self.parent = parent
        if children: # Set children only if they are not None
            self.children = children

class Tree:
    root: TreeNode
    n_nodes: int

    def __init__(self, root: TreeNode, n_nodes: int) -> None:
        self.root = root
        self.n_nodes = n_nodes

    @classmethod
    def from_lines(cls, lines: list[str]):
        assert ('[TREE' in lines[0])
        root = None
        n_nodes = int(lines[1].split(': ')[1])
        parents = {}
        n = 0
        while n < n_nodes:
            line = lines[n + 2]
            idx, ntype, left, right, feature, thr, _, klass = line.split()
            feature = int(feature)
            thr = float(thr)
            klass = int(klass)
            node = TreeNode(
                name=idx,
                feature=feature,
                thr=thr,
                klass=klass
            )
            if node.name in parents:
                node.parent = parents[node.name]
            else:
                root = node
            if ntype == 'IN':
                parents[left] = node
                parents[right] = node
            n += 1
        return cls(root, n_nodes)

    def getClass(self, x: pd.Series):
        node = self.root
        while node.klass == -1:
            if x[node.feature] <= node.thr:
                node = node.children[0]
            else:
                node = node.children[1]
        return node.klass

    def getF(self, x: pd.Series, c: int):
        return 1.0 if self.getClass(x) == c else 0.0

class TreeEnsemble:
    def __init__(
        self,
        dataset: str = 'train.csv',
        etype: str = 'RF',
        n_classes: int = 0,
        n_features: int = 0,
        m_depth: int = 0,
        trees: None | list[Tree] = None,
        weigths: None | np.ndarray = None
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

    @classmethod
    def from_file(cls, file: str):
        with open(file, 'r') as f:
            lines = f.readlines()
            f.close()

        dataset = lines[0].split(': ')[1]
        etype = lines[1].split(': ')[1]
        n_trees = int(lines[2].split(': ')[1])
        n_features = int(lines[3].split(': ')[1])
        n_classes = int(lines[4].split(': ')[1])
        m_depth = int(lines[5].split(': ')[1])
        t = 0
        lineIdx = 8
        trees = []
        while t < n_trees:
            n_nodes = int(lines[lineIdx + 1].split(': ')[1])
            tree = Tree.from_lines(lines[lineIdx:])
            trees.append(tree)
            lineIdx += (n_nodes + 3)
            t += 1
        return cls(
            dataset=dataset,
            etype=etype,
            n_classes=n_classes,
            n_features=n_features,
            m_depth=m_depth,
            trees=trees
        )
    