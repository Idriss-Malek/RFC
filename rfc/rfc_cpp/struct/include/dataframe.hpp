#pragma once

#include <iostream>
#include <vector>
#include <variant>
#include "feature.hpp"

using namespace std;

class DataFrame {

public:
    vector<vector<double>> data;
    vector<Feature> features;

    DataFrame(vector<vector<double>> data);
    DataFrame();

    vector<double> getRow(int index);
    void addRow(vector<double>);
    

};

DataFrame read_csv(std::string file);