#pragma once

#include <iostream>
#include <vector>
#include <variant>

using namespace std;

class DataFrame {

public:
    vector<vector<double>> data;

    DataFrame(vector<vector<double>> data);

    vector<double> getRow(int index);

};