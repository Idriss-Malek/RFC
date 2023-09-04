#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include <Eigen/Dense>

#include "feature.hpp"
#include "dataframe.hpp"
#include "node.hpp"

class Tree {
public:
    int id;
    Node* root;
    std::string name;
    int n_classes;

    Tree(int id_, Node* root_, int n_classes, std::string name_);

    Node* leaf(std::vector <double> x);
    int klass(std::vector<double> x);
    float F(std::vector<double> x, int klass_);
    Eigen::MatrixXi full_distribution(DataFrame df);


};