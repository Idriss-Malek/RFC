#pragma once

#include <iostream>
#include <vector>
#include <variant>

struct Numpy{
        std::vector<int> array;
        int size();
        int& operator[](int index);
    };
Numpy operator+(const Numpy& a1, const Numpy& a2);

class Prob{
public:
    std::vector<Numpy> probs;

    Prob update(Numpy a);

};
