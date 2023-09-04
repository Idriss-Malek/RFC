#include <iostream>
#include <vector>
#include <variant>
#include "../include/feature.hpp"
#include "../include/tree.hpp"

Node::Node(int id_,Feature* feature_ = nullptr, float thr_ = -1, int klass_ = -1, std::string name_ = "", Node* left_ = nullptr,Node* right_ = nullptr ){
    this->id = id_;
    this->name = name_.empty() ? "Node " + std::to_string(id_) : name;
    this->feature = feature_;
    this->thr = thr_;
    this->klass = klass_;
    this->left=left_;
    this->right = right_;
}

Node::Node(int id_){
    this->id = id_;
}

void Node::setLeft(Node* left_){
    this->left = left_;
}

void Node::setRight(Node* right_){
    this->right = right_;
}

void Node::setParent(Node* parent_){
    this->parent = parent_;
}

Node* Node::getLeft(){
    return this->left;
}

Node* Node::getRight(){
    return this->right;
}

Node* Node::getParent(){
    return this->parent;
}

bool Node::is_leaf(){
    if (klass != -1){
        return true;
    }
    else{
        return false;
    }
}

bool Node::goes_to_left(double value){
    if (!this->is_leaf()){
        if (this->feature->isnumerical()){
            return static_cast<float>(value) <= this->thr;
        }
        if (this->feature->isbinary()){
            return static_cast<int>(value) == 0;
        }
    }
}

float Node::p(int klass_){
    return 0.0f+(this->klass == klass_);
}

Node* Node::split(double value){
    if (goes_to_left(value)){
        return this->left;
    }
    else{
        return this->right;
    }
}