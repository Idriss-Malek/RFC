#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include "feature.hpp"

class Node {
public:
    int id;
    Feature* feature;
    float thr;
    int klass;
    std::string name;
    std::vector<int> categories;
    Node* left;
    Node* right;
    Node* parent;

    Node(int id_,Feature* feature_, float thr_, int klass_ , std::string name_, std::vector<int> categories_, Node* left_ ,Node* right_,Node* parent_);
    
    void setLeft(Node* left_);
    void setRight(Node* right_);
    void setParent(Node* parent_);
    
    Node* getLeft();
    Node* getRight();
    Node* getParent();
    
    bool is_leaf();
    float p(int klass_);
    bool goes_to_left(double value);
    Node* split(double value);

};