#pragma once
#include <algorithm>
#include <iostream>
#include <vector>
#include <variant>
#include "../include/tree.hpp"
#include "../include/dataframe.hpp"
#include "../include/prob.hpp"
#include "../include/forest.hpp"

Forest::Forest(std::vector<Tree> trees, std::vector<double> weights, int n_classes){
    this->trees = trees;
    this->weights = weights;
    this->n_classes = n_classes;
}

void increment_recorder(std::vector<int> recorder, std::vector<double> x, Tree tree){
    int klass = tree.klass(x);
    recorder[klass] = recorder[klass] + 1;
}

void Forest::distribution(std::vector<double> x, std::vector<Numpy> prob){
    std::vector<int> res(this->n_classes);
    res.assign(this->n_classes,0);
    std::for_each(this->trees.begin(), this->trees.end(), [res,x](Tree tree) {return increment_recorder(res, x , tree);});
    Numpy cur(res);
    prob.push_back(cur);
}


Prob Forest::full_distribution(DataFrame df){
    std::vector<Numpy> res;
    std::for_each(df.data.begin(), df.data.end(),[this,res](std::vector<double> x) {return this->distribution(x , res);} );
    Prob cur(res);
    return cur;
}