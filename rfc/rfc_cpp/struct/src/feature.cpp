#include <iostream>
#include <vector>
#include <variant>
#include "../include/feature.hpp"

Feature::Feature(int id_, FeatureType type_, std::string name = "") {
    this->id = id_;
    this->name = name.empty() ? "Feature " + std::to_string(id_) : name;
    this->type = type_;
}

bool Feature::isnumerical() {
    return this->type == FeatureType::NUMERICAL;
}

bool Feature::isbinary() {
    return this->type == FeatureType::BINARY;
}

bool Feature::iscategorical() {
    return this->type == FeatureType::CATEGORICAL;
}

double Feature::value(std::vector<double> x){
    return x[this->id];
}
/*
void Feature::setLevels(std::vector<float> levels_){
    if (!(this->isnumerical())){
        throw std::invalid_argument("Levels are only defined for numerical features!");
    }
    this->levels = levels_;
}

std::vector<float> Feature::getLevels(){
    if (!(this->isnumerical())){
        throw std::invalid_argument("Levels are only defined for numerical features!");
    }

    return this->levels;
}

std::vector<int> Feature::getCategories(){
    if (!(this->iscategorical())){
        throw std::invalid_argument("Categories are only defined for categorical features!");
    }
    return this->categories;
}

void Feature::setCategories(std::vector<int> categories_){
    if (!(this->iscategorical())){
        throw std::invalid_argument("Categories are only defined for categorical features!");
    }
    this->categories = categories_;
}
*/

bool Feature::operator==(const Feature& other) const {
    return this->id == other.id;
}



namespace std {
    template<>
    struct hash<Feature> {
        size_t operator()(const Feature& obj) const {
            return std::hash<int>()(obj.id); // Use the hash function for int
        }
    };
}
/*
int main() {
    FeatureType type1 = FeatureType::NUMERICAL;
    Feature feature1(1,type1,"ftr1");
    std::cout<<feature1<<std::endl;
    std::cout<<feature1.isnumerical()<<std::endl;
    std::vector<float> nums = {1.f, 2.f, 3.14f, 4.f, 5.f};

    std::cout<<std::endl;
    Feature feature2(2,type1,"ftr2");
    std::cout<<(feature1==feature2)<<std::endl;
    std::cout<<std::hash<Feature>{}(feature1)<<std::endl;

}
*/