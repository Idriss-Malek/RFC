import numpy as np

from collections.abc import Iterable, Iterator

from enum import Enum

from .feature import Feature, FeatureType
from .node import Node
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

    @classmethod
    def from_file(cls, file: str, log_output: bool = False) -> "TreeEnsemble":
        with open(file, "r") as f:
            lines = f.readlines()
            f.close()

        # dataset = lines[0].split(": ")[1]
        etype = lines[1].strip().split(": ")[1]
        n_trees = int(lines[2].strip().split(": ")[1])
        n_features = int(lines[3].strip().split(": ")[1])
        n_classes = int(lines[4].strip().split(": ")[1])
        # m_depth = int(lines[5].split(": ")[1])
        
        lineIdx = 8

        if log_output:
            print(f"Loading {n_trees} trees with {n_features} features and {n_classes} classes.")
        
        lineIdx += 1
        features: list[Feature] = []
        for f in range(n_features):
            if log_output:
                print(f"Loading feature {f+1}/{n_features}.")
            line = lines[lineIdx]
            name, ftype = line.strip().split(": ")
            match ftype:
                case 'F':
                    ftype = FeatureType.NUMERICAL
                case 'D':
                    ftype = FeatureType.NUMERICAL
                    lineIdx += 1 # TODO: remove this line.
                case 'C':
                    ftype = FeatureType.CATEGORICAL
                case 'B':
                    ftype = FeatureType.BINARY
                case _:
                    raise ValueError(f"Unknown feature type: {ftype}")
            ftype = FeatureType(ftype)
            feature = Feature(
                id_=f,
                name=name,
                ftype=ftype
            )
            if ftype == FeatureType.CATEGORICAL:
                lineIdx += 1
                line = lines[lineIdx]
                categories = line.strip().split()
                feature.categories = categories
            features.append(feature)
            lineIdx += 1
        
        lineIdx += 1

        trees = []
        t = 0
        while t < n_trees:
            if log_output:
                print(f"Loading tree {t+1}/{n_trees}.")
            lineIdx += 1
            line = lines[lineIdx]
            n_nodes = int(line.strip().split(": ")[1])
            if log_output:
                print(f"Loading {n_nodes} nodes.")

            lineIdx += 1
            parents: dict[int, tuple[Node, ChildType]] = {}
            nodes: dict[int, Node] = {}
            n = 0
            while n < n_nodes:
                if log_output:
                    print(f"Loading node {n+1}/{n_nodes}.")
                line = lines[lineIdx + n]
                nodeId, ntype, leftId, rightId, _, _, _, _ = line.strip().split()
                nodeId = int(nodeId)
                ntype = NodeType(ntype)
                leftId = int(leftId)
                rightId = int(rightId)
                node = Node(id_=nodeId)
                nodes[node.id] = node
                if ntype == NodeType.INTERNAL:
                    parents[leftId] = (node, ChildType.LEFT)
                    parents[rightId] = (node, ChildType.RIGHT)
                if node.id in parents:
                    parent, side = parents[node.id]
                    match side:
                        case ChildType.LEFT: parent.left = node
                        case ChildType.RIGHT: parent.right = node
                n += 1

            n = 0
            while n < n_nodes:
                line = lines[lineIdx + n]
                nodeId, ntype, _, _, featureId, val, _, klass = line.strip().split()
                nodeId = int(nodeId)
                ntype = NodeType(ntype)
                featureId = int(featureId)
                klass = int(klass)
                node = nodes[nodeId]
                match ntype:
                    case NodeType.LEAF:
                        node.klass = klass
                    case NodeType.INTERNAL:
                        feature = features[featureId]
                        node.feature = feature
                        match feature.ftype:
                            case FeatureType.CATEGORICAL:
                                node.categories = [val]
                            case FeatureType.NUMERICAL:
                                node.threshold = float(val)
                n += 1

            root = nodes[0]
            tree = Tree(id_ = t, root = root)
            trees.append(tree)
            lineIdx += n_nodes
            lineIdx += 1
            t += 1

        if log_output:
            print("Done!")
    
    
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

    def __getitem__(self, idx: int) -> Tree:
        return self.trees[idx]

    def __iter__(self) -> Iterator[Tree]:
        return self.trees.__iter__()

    def __len__(self) -> int:
        return len(self.trees)