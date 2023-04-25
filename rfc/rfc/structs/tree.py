import numpy as np

from anytree import PreOrderIter
from collections.abc import Iterable, Iterator, Callable

from .node import Node

class Tree(Iterable[Node]):
    id: int
    name: str
    root: Node

    def __init__(
        self,
        id_: int,
        root: Node,
        name: None | str = None,
    ) -> None:
        self.id = id_
        self.name = name if name is not None else f'Tree {id_}'
        self.root = root

    def getLeaf(self, x: np.ndarray) -> Node:
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
        filter_: None | Callable[[Node], bool] = None,
    ) -> Iterator[Node]:
        def filter__(node: Node) -> bool:
            return not node.is_leaf and (filter_ is None or filter_(node))
        return PreOrderIter(self.root, filter_=filter__)

    def getProbas(self, c: int):
        return np.array([leaf.klass == c for leaf in self.getLeaves()], dtype=np.float32)

    def getLeaves(self) -> Iterator[Node]:
        return self.root.leaves.__iter__()

    def getNodesAtDepth(
        self,
        depth: int
    ) -> Iterator[Node]:
        return self.getNodes(lambda node: node.depth == depth)

    def getNodesWithFeature(
        self,
        feature: int
    ) -> Iterator[Node]:
        return self.getNodes(lambda node: node.feature == feature)

    def __iter__(self) -> Iterator[Node]:
        return PreOrderIter(self.root)

    def __len__(self) -> int:
        return len(self.root.descendants) + 1

    @property
    def depth(self) -> int:
        return self.root.height