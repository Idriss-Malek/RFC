from reader import Reader
import cplex
import pandas as pd


class Optimizer:
    def __init__(self, rf_file, dataset):
        self.rf = Reader(rf_file)
        self.dataset = pd.read_csv(dataset)

    def opt(self):
        model = cplex.Cplex()
        u = model.variables.add(names=[f"u{i}" for i in range(self.rf.nb_trees)],
                                types=[model.variables.type.binary for i in range(self.rf.nb_trees)])
        constraints = []
        original_rf_class = self.rf.rf_decision(x)
        if self.rf.nb_classes == 2:
            for index, row in self.dataset.iterrows():
                constraints.append([[[f"u{i}" for i in range(self.rf.nb_trees)],
                                     [self.rf.tree_fun(row, self.rf.trees[i], original_rf_class) - self.rf.tree_fun(row,
                                                                                                                    self.rf.trees[
                                                                                                                        i],
                                                                                                                    1 - original_rf_class)
                                      for i in range(self.rf.nb_trees)]], ">=", 0.0])
        else:
            for index, row in self.dataset.iterrows():
                for c in self.rf.nb_classes:
                    constraints.append([[[f"u{i}" for i in range(self.rf.nb_trees)],
                                         [self.rf.tree_fun(row, self.rf.trees[i], original_rf_class) - self.rf.tree_fun(
                                             row,
                                             self.rf.trees[
                                                 i],
                                             c)
                                          for i in range(self.rf.nb_trees)]], ">=", 0.0])
        model.linear_constraints.add(lin_expr=constraints)
        model.objective.set_sense(model.objective.sense.minimize)
        model.objective.set_linear([(f"u{i}", 1.0) for i in range(self.rf.nb_trees)])
        model.solve()
