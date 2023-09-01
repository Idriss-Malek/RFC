#include <iostream>
#include <vector>
#include <variant>
#include <fstream>
#include <sstream>
#include "../include/dataframe.hpp"
#include "../include/feature.hpp"
#include "../include/feature_type.hpp"

DataFrame::DataFrame(std::vector<std::vector<double>> data){
    this->data = data;
}

DataFrame::DataFrame(){
    std::vector<std::vector<double>> data;
    this->data = data;
}

std::vector<double> DataFrame::getRow(int index){
    return this->data[index];
}

void DataFrame::addRow(std::vector<double> row){
    this->data.push_back(row);
}

DataFrame read_csv(std::string file_){
    std::ifstream file(file_);
    DataFrame data;
    std::string line;
    std::vector<std::string> names;
    std::vector<Feature> features;
    int line_nbr = 0;
    while (std::getline(file, line)) {
        if (line_nbr == 0){
            line_nbr ++;
            std::istringstream ss(line);
            std::string field;
            while (std::getline(ss, field, ',')) {
                names.push_back(field);
            }
            }

        
        else if (line_nbr == 1){
            line_nbr++;
            std::istringstream ss(line);
            std::string field;
            int i = 0;
            while (std::getline(ss, field, ',')) {
                FeatureType type_;
                if (field == "F" || field == "D"){
                    type_ = FeatureType::NUMERICAL;
                }
                if (field == "B"){
                    type_ = FeatureType::BINARY;
                }
                Feature feat(i,type_,names[i]); 
                i++;
                features.push_back(feat);
            }
            data.features = features;

        }
        else {std::vector<double> fields;
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(std::stod(field));
        }
        data.addRow(fields);}
    }
    return data;
}

int main(){
    DataFrame data = read_csv("test.csv");
    for (int i = 0; i < data.features.size(); ++i) {
        std::cout << data.features[i] << std::endl;
    }
    for (int i = 0; i < data.data.size(); ++i) {
        for (int j = 0; j< data.data[i].size(); ++j) {
            std::cout << data.data[i][j] << " ";
        }
        std::cout<<std::endl;
    }
    return 0;
}