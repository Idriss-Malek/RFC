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

    Node(int id_,Feature* feature_, float thr_, int klass_ , std::string name_, Node* left_ ,Node* right_);
    Node(int id_);

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

    friend std::ostream& operator<<(std::ostream& os, Node& obj) {
        std::string leaf = obj.is_leaf() ? "Leaf" : "Internal";
        std::string feature = !obj.is_leaf() ? obj.feature->name : "None";
        std::string thr = !obj.is_leaf() ? std::to_string(obj.thr) : "None";
        std::string klass = obj.is_leaf() ? std::to_string(obj.klass) : "None";

        os << leaf<<" Node : " << obj.id << " Feature "<<feature << " Thr " << thr<< " Klass " << klass <<std::endl;
        return os;
    }

};