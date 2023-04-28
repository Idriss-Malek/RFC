from enum import Enum

from ..structs.ensemble import Ensemble
from ..structs.node import Node
from ..structs.feature import Feature, FeatureType
from ..structs.tree import Tree

class NodeType(Enum):
    INTERNAL = 'IN'
    LEAF = 'LN'

class ChildType(Enum):
    LEFT = 'L'
    RIGHT = 'R'

def load_tree_ensemble(file: str, log_output: bool = False) -> Ensemble:
    with open(file, "r") as f:
        lines = f.readlines()
        f.close()

    # dataset = lines[0].split(": ")[1]
    etype = lines[1].strip().split(": ")[1]
    n_trees = int(lines[2].strip().split(": ")[1])
    n_features = int(lines[3].strip().split(": ")[1])
    n_classes = int(lines[4].strip().split(": ")[1])
    # m_depth = int(lines[5].split(": ")[1])
    
    idx = 8

    if log_output: print(f"Loading {n_trees} trees with {n_features} features and {n_classes} classes.")
    
    idx += 1
    features: list[Feature] = []
    for f in range(n_features):
        if log_output: print(f"Loading feature {f+1}/{n_features}.")
        line = lines[idx]
        featureName, featureType = line.strip().split(": ")
        match featureType:
            case 'F':
                featureType = FeatureType.NUMERICAL
            case 'D':
                featureType = FeatureType.NUMERICAL
            case 'C':
                featureType = FeatureType.CATEGORICAL
            case 'B':
                featureType = FeatureType.BINARY
            case _:
                raise ValueError(f"Unknown feature type: {featureType}")
        featureName = featureName.strip()
        featureType = FeatureType(featureType)
        feature = Feature(
            id_=f,
            name=featureName,
            type_=featureType
        )
        if featureType == FeatureType.CATEGORICAL:
            idx += 1
            line = lines[idx]
            categories = line.strip().split()
            feature.categories = categories
        features.append(feature)
        idx += 1
    
    idx += 1
    trees = []
    t = 0
    while t < n_trees:
        if log_output: print(f"Loading tree {t+1}/{n_trees}.")
        idx += 1
        line = lines[idx]
        n_nodes = int(line.strip().split(": ")[1])
        if log_output: print(f"Loading {n_nodes} nodes.")

        idx += 1
        parents: dict[int, tuple[Node, ChildType]] = {}
        nodes: dict[int, Node] = {}
        n = 0
        while n < n_nodes:
            if log_output: print(f"Loading node {n+1}/{n_nodes}.")
            line = lines[idx + n]
            nodeId, nodeType, leftId, rightId, _, _, _, _ = line.strip().split()
            nodeId = int(nodeId)
            nodeType = NodeType(nodeType)
            leftId = int(leftId)
            rightId = int(rightId)
            node = Node(id_=nodeId)
            nodes[node.id] = node
            if nodeType == NodeType.INTERNAL:
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
            line = lines[idx + n]
            nodeId, nodeType, _, _, featureId, val, _, klass = line.strip().split()
            nodeId = int(nodeId)
            nodeType = NodeType(nodeType)
            featureId = int(featureId)
            klass = int(klass)
            node = nodes[nodeId]
            match nodeType:
                case NodeType.LEAF:
                    node.klass = klass
                case NodeType.INTERNAL:
                    feature = features[featureId]
                    node.feature = feature
                    match feature.type:
                        case FeatureType.CATEGORICAL: node.categories = [val]
                        case FeatureType.NUMERICAL: node.threshold = float(val)
                        case _: pass
            n += 1

        root = nodes[0]
        T = Tree(id_ = t, root = root)
        trees.append(T)
        idx += n_nodes
        idx += 1
        t += 1

    if log_output:
        print("Done!")

    return Ensemble(
        trees=trees,
        features=features,
        n_classes=n_classes,
        type_=etype
    )