#include <iostream>
#include <vector>
#include <variant>
#include "dataframe.hpp"

DataFrame::DataFrame(std::vector<std::vector<double>> data){
    this->data = data;
}

std::vector<double> DataFrame::getRow(int index){
    return this->data[index];
}