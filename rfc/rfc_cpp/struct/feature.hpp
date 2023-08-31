#pragma once
#include <variant>
#include <iostream>
#include <vector>
#include "feature_type.hpp"

class IdentifiedObject {
public:
    int id;
};

class Feature {
public:
    int id;
    std::string name;
    FeatureType type;
    std::vector<float> levels;
    std::vector<int> categories;

    Feature(int id_, FeatureType type_, std::string name, std::vector<float> levels, std::vector<int> categories);

    bool isnumerical();

    bool isbinary();

    bool iscategorical();

    double value(std::vector<double> x);

    void setLevels(std::vector<float> levels_);

    std::vector<float> getLevels();

    std::vector<int> getCategories();
    
    void setCategories(std::vector<int> categories_);

    friend std::ostream& operator<<(std::ostream& os, const Feature& obj) {
    os << "Feature : " << obj.name;
    return os;
    }

    bool operator==(const Feature& other) const;

};

