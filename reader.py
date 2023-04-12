import numpy as np


class Reader:
    def __init__(self, rf_file):
        self.rf_file = rf_file
        trees = []
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
                    trees[-1].append(line.strip())
                    trees[-1][-1][5] = float(trees[-1][-1][5])
                    for i in range(8):
                        if i != 1 and i != 5:
                            trees[-1][-1][i] = int(trees[-1][-1][i])
                if '[TREE' in line:
                    trees.append([])
                else:
                    next(f)

        def aux(x):
            tree_results = np.empty(self.nb_trees)
            for i in range(self.nb_trees):
                node = trees[i][0]
                while node[1] == 'IN':
                    if x[node[4]] <= node[5]:
                        node = trees[i][node[2]]
                    else:
                        node = trees[i][node[3]]
                tree_results[i] = node[-1]
            return np.argmax(np.bincount(tree_results))

        self.fun = aux

