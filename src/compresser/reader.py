import numpy as np


class Reader:
    def __init__(self, rf_file):
        self.rf_file = rf_file
        self.trees = []
        with open(self.rf_file, 'r') as f:
            self.dataset = f.readline().split()[1]
            next(f)
            self.nb_trees = int(f.readline().strip()[1])
            self.nb_features = int(f.readline().strip()[1])
            self.nb_classes = int(f.readline().strip()[1])
            self.max_depth = int(f.readline().strip()[1])
            next(f)
            numbers = [str(i) for i in range(10)]

            for line in f:
                if line[0] in numbers:
                    self.trees[-1].append(line.strip())
                    self.trees[-1][-1][5] = float(self.trees[-1][-1][5])
                    for i in range(8):
                        if i != 1 and i != 5:
                            self.trees[-1][-1][i] = int(self.trees[-1][-1][i])
                if '[TREE' in line:
                    self.trees.append([])
                else:
                    next(f)

    def tree_fun(self, x, tree, c):
        node = tree[0]
        while node[1] == 'IN':
            if x[node[4]] <= node[5]:
                node = tree[node[2]]
            else:
                node = tree[node[3]]
        return c == node[-1] + 1.

    def rf_fun(self, x, c):
        tree_results = np.empty(self.nb_trees)
        for i in range(self.nb_trees):
            tree_results[i] = self.tree_fun(x, self.trees[i], c)
        return np.mean(tree_results)

    def rf_decision(self, x, array=None):
        if array is None:
            array = np.empty(self.nb_classes)
        for c in range(self.nb_classes):
            array[c] = self.rf_fun(x, c)
        return np.argmax(array)
