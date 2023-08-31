#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include "feature.hpp"
#include "node.hpp"
#include "tree.hpp"

Tree::Tree(int id_, Node* root_, std::string name_){
    this->id = id_;
    this->name = name_.empty() ? "Tree " + std::to_string(id_) : name;
    this->root = root_;
}

Node* Tree::leaf(std::vector <double> x){
    Node* cur = this->root;
    while (!(cur->is_leaf())){
        cur = cur->split(x[cur->feature->id]);
    }
    return cur;
}

int Tree::klass(std::vector<double> x){
    return this->leaf(x)->klass;
}

float Tree::F(std::vector<double> x, int klass_){
    return 0.0f+(klass_ == this->klass(x));
}