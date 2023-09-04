#pragma once

#include "forest.hpp"
#include "dataframe.hpp"
#include <Eigen/Dense>

Eigen::VectorXi probs_to_class(Eigen::MatrixXi probs);

class Greedy{
public:
    Forest rf;
    DataFrame df;
    Eigen::MatrixXi def_probs;
    Eigen::VectorXi def_klass;
    std::vector<int> u;

    Greedy(DataFrame df_, Forest forest_);

    void compress(int iter = 100);
};

