from ..structs.ensemble import TreeEnsemble
from ..structs.node import Node
from ..structs.feature import Feature, FeatureType
from ..structs.tree import Tree
from enum import Enum

class NodeType(Enum):
    INTERNAL = 'IN'
    LEAF = 'LN'

class ChildType(Enum):
    LEFT = 'L'
    RIGHT = 'R'

def load_tree_ensemble(file: str, log_output: bool = False) -> TreeEnsemble:
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

    return TreeEnsemble(
        trees=trees,
        features=features,
        n_classes=n_classes,
        etype=etype
    )