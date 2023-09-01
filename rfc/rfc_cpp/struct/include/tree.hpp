#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include "feature.hpp"
#include "node.hpp"

class Tree {
public:
    int id;
    Node* root;
    std::string name;

    Tree(int id_, Node* root_, std::string name_);

    Node* leaf(std::vector <double> x);
    int klass(std::vector<double> x);
    float F(std::vector<double> x, int klass_);


};