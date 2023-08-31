#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include "prob.hpp"

int Numpy::size(){
    return this->array.size();
} 
int& Numpy::operator[](int index){
    return this->array[index];
}
Numpy operator+(const Numpy& a1, const Numpy& a2){
    std::vector<int> res; 
    for (int i = 0; i < a1.size(); i++) {
        res.push_back(a1[i]+a2[i])
}
}

