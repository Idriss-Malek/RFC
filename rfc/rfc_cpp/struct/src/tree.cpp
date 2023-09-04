#include <iostream>
#include <vector>
#include <variant>
#include <Eigen/Dense>

#include "../include/feature.hpp"
#include "../include/node.hpp"
#include "../include/tree.hpp"
#include "../include/dataframe.hpp"

Tree::Tree(int id_, Node* root_, int n_classes, std::string name_){
    this->id = id_;
    this->name = name_.empty() ? "Tree " + std::to_string(id_) : name;
    this->root = root_;
    this->n_classes = n_classes;
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

Eigen::MatrixXi Tree::full_distribution(DataFrame df){
    Eigen::MatrixXi m(df.data.size(),this->n_classes);
    Eigen::VectorXi v(this->n_classes);
    for (int i=0; i < df.data.size(); i++){
        v.fill(0);
        v[this->klass(df[i])] = 1;
        m.row(i) = v;
    }
    return m;
}

