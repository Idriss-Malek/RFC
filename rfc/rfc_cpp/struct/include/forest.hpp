#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
#include <variant>
#include <fstream>
#include <sstream>
#include "tree.hpp"
#include "dataframe.hpp"
#include "forest.hpp"

class Forest{
public:
    std::vector<Tree> trees;
    int n_classes;
    DataFrame df;
    std::vector<double> weights;
    std::vector<Feature> features;

    Forest(std::vector<Tree> trees, std::vector<double> weights, int n_classes);
    Forest(std::string rf);

    Eigen::VectorXi distribution(std::vector<double> x);

    Eigen::MatrixXi full_distribution(DataFrame df);
    Eigen::VectorXi klass(DataFrame df);

};

