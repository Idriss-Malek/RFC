#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include "tree.hpp"
#include "dataframe.hpp"
#include "prob.hpp"

class Forest{
public:
    std::vector<Tree> trees;
    std::vector<double> weights;
    int n_classes;

    Forest(std::vector<Tree> trees, std::vector<double> weights, int n_classes);

    void distribution(std::vector<double> x, std::vector<Numpy> prob);

    Prob full_distribution(DataFrame df);
    std::vector<int> klass(DataFrame df);

};