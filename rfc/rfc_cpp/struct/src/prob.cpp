#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include "../include/prob.hpp"

Numpy::Numpy(std::vector<int> array){
    this->array = array;
}

int Numpy::size(){
    return this->array.size();
} 
int& Numpy::operator[](int index){
    return this->array[index];
}
Numpy operator+( Numpy& a1,  Numpy& a2){
    std::vector<int> res; 
    for (int i = 0; i < a1.size(); i++) {
        res.push_back(a1[i]+a2[i]);
        }
    return Numpy(res);

}
Numpy operator-( Numpy& a1,  Numpy& a2){
    std::vector<int> res; 
    for (int i = 0; i < a1.size(); i++) {
        res.push_back(a1[i]-a2[i]);
        }
    return Numpy(res);

}

Prob::Prob(std::vector<Numpy> probs){
    this->probs = probs;
}

void Prob::update(Numpy a){
    for (int i = 0; i < this->probs.size(); i++) {
        this->probs[i] = this->probs[i] - a;
    }
}

