#include <iostream>
#include <vector>
#include <variant>

enum class FeatureType {
    NUMERICAL = 0,
    CATEGORICAL = 1,
    BINARY = 2
};

class IdentifiedObject {
public:
    int id;
};

class Feature {
public:
    std::string name;
    FeatureType type;
    std::vector<float> levels;
    std::vector<std::variant<std::monostate, float, std::string>> categories;

    Feature(int id_, FeatureType type_, std::string name = "", std::vector<float> levels = {}, std::vector<std::variant<std::monostate, float, std::string>> categories = {}) {
        this->id = id_;
        this->name = name.empty() ? "Feature " + std::to_string(id_) : name;
        this->type = type_;
        this->levels = levels;
        this->categories = categories;
    }

    bool isnumerical() {
        return this->type == FeatureType::NUMERICAL;
    }

    bool isbinary() {
        return this->type == FeatureType::BINARY;
    }

    bool iscategorical() {
        return this->type == FeatureType::CATEGORICAL;
    }

    std::variant<std::monostate, float, std::string> value(std::vector<std::variant<std::monostate, float, std::string>> x) {
        return x[this->id];
    }
};
