# -*- coding: utf-8 -*-

from anytree import PreOrderIter, RenderTree
from collections.abc import Iterable, Iterator, Callable

from ..types import *
from .feature import Feature
from .node import Node

class Tree(IdentifiedObject, Iterable[Node]):
    name: str
    root: Node

    """
    A class that defines a decision tree.
    
    The tree is represented as a binary tree, where each node is either 
    a leaf node or an internal node.

    To contruct a tree, you need to specify the root node, the id and the name.
    
    >>> from rfc.structs import Node, Tree
    >>> root = Node(0, name='Root')
    >>> tree = Tree(0, root, name='Tree')
    """

    def __init__(
        self,
        id_: int,
        root: Node,
        name: None | str = None,
    ) -> None:
        """
        
        """
        self.id = id_
        self.name = name if name is not None else f'Tree {id_}'
        self.root = root

    def leaf(self, x: Sample) -> Node:
        """
        Returns the leaf node that the sample x falls into.

        >>> x = np.array([1, 0, 1])
        >>> tree.leaf(x)
        Args:
            x (Sample): The sample to be classified.

        Returns:
            Node: The leaf node that the sample x falls into.
        """
        node = self.root
        while not node.is_leaf:
            node = node.split(x)
        return node

    def klass(self, x: Sample) -> int:
        """
        Returns the class of the sample x.

        Args:
            x (Sample): The sample to be classified.

        Returns:
            int: The class of the sample x.
        """
        return self.leaf(x).klass
    
    def path(self, x: Sample):
        path = [self.root]
        while not path[-1].is_leaf:
            path.append(path[-1].split(x))
        return path


    def F(self, x: Sample, c: int):
        """
        Returns 1 if the sample x belongs to class c, 0 otherwise.

        Args:
            x (Sample): The sample to be classified.
            c (int): The class to be checked.

        Returns:
            float: 1 if the sample x belongs to class c, 0 otherwise.
        """
        return 1.0 if self.klass(x) == c else 0.0

    def p(self, c: int) -> list[float]:
        """
        Returns the class probabilities of the class c.

        Args:
            c (int): The class to be checked.

        Returns:
            list[float]: The class probabilities of the class c.
        """
        return [leaf.p(c) for leaf in self.leaves]

    def nodes_at_depth(self, depth: int) -> tuple[Node, ...]:
        """
        Returns all internal nodes at the specified depth.

        Args:
            depth (int): The depth of the nodes to be returned.

        Returns:
            tuple[Node, ...]: All internal nodes at the specified depth.
        """
        return tuple(self.__nodes(lambda node: node.depth == depth))

    def nodes_with_feature(self, f: int | Feature) -> tuple[Node, ...]:
        """
        Returns all internal nodes that split on the specified feature.

        Args:
            f (int | Feature): The feature to be checked or its id.

        Returns:
            tuple[Node, ...]: All internal nodes that split on the specified feature.
        """
        return tuple(self.__nodes(lambda node: node.split_on(f)))
    
    def nodes_with_feature_and_level(self, f: int | Feature, level: float) -> tuple[Node, ...]:
        """
        Returns all internal nodes that split on the specified numerical feature with the specified level.

        Args:
            f (int | Feature): The feature to be checked or its id.
            level (float): The level to be checked.

        Returns:
            tuple[Node, ...]: All internal nodes that split on the specified numerical feature with the specified level.
        """
        return tuple(self.__nodes(lambda node: (node.split_on(f) and node.threshold == level)))

    @property
    def nodes(self) -> tuple[Node, ...]:
        """
        Returns all intrenal nodes of the tree.
        
        Returns:
            tuple[Node, ...]: All internal nodes of the tree.
        """
        return tuple(self.__nodes())

    @property
    def leaves(self) -> tuple[Node, ...]:
        """
        Returns all leaf nodes of the tree.
        
        Returns:
            tuple[Node, ...]: All leaf nodes of the tree.
        """
        return self.root.leaves

    @property
    def depth(self) -> int:
        """
        Returns the maximum depth of the tree.

        Returns:
            int: The maximum depth of the tree.
        """
        return self.root.height

    def __iter__(self) -> Iterator[Node]:
        return PreOrderIter(self.root)

    def __len__(self) -> int:
        return len(self.root.descendants) + 1

    def __repr__(self) -> str:
        return f'Tree: id={self.id}, name={self.name}\n{RenderTree(self.root)}'

    def __nodes(
        self,
        filter_: None | Callable[[Node], bool] = None,
    ) -> Iterator[Node]:
        def filter__(node: Node) -> bool:
            return not node.is_leaf and (filter_ is None or filter_(node))
        return PreOrderIter(self.root, filter_=filter__)
    
    def new_tree(self,u):
        s
        if u[self.root.id] == 1:
            f=self.root.id
            for v in PreOrderIter(f)[1:]:
                if u[v.id] == 1:
                    a
        else:
            return None