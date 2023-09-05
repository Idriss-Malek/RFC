#include <algorithm>
#include <iostream>
#include <vector>
#include <variant>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "../include/tree.hpp"
#include "../include/dataframe.hpp"
#include "../include/forest.hpp"
#include <Eigen/Dense>

Forest::Forest(std::vector<Tree> trees, std::vector<double> weights, int n_classes){
    this->trees = trees;
    this->weights = weights;
    this->n_classes = n_classes;
}

Forest::Forest(){}

Forest::Forest(std::string rf ){
    std::vector<Feature> features;
    int n_classes;
    std::ifstream file(rf);
    bool read = false;
    bool feat = false;
    int feat_nbr=0;
    std::vector<Tree> trees;
    static std::vector<std::vector<Node>> nodes;
    std::string line;
    std::vector<Node> mini_nodes;
    int n_tree = -1;
    while (std::getline(file, line)) {   
        if (read){
            if (line == ""){
                read=false;
                trees.push_back(Tree(n_tree,&nodes.back()[0],n_classes,""));
                continue;
            }
            std::vector<std::string> node_str;
            std::istringstream ss(line);
            std::string field;
            while (std::getline(ss, field, ' ')) {
                node_str.push_back(field);
            }
            int id_ = std::stoi(node_str[0]);
            Feature* feature_ = (node_str[4] == "-1") ? nullptr : &(this->features[std::stoi(node_str[4])]);
            float thr = std::stof(node_str[5]);
            int klass = std::stoi(node_str[7]);
            Node* left = &(nodes.back()[std::stoi(node_str[2])]);
            Node* right = &(nodes.back()[std::stoi(node_str[3])]);
            nodes.back()[id_] = Node(id_, feature_, thr, klass,std::string(""), left, right);

        }
        else if (line.substr(0, 10) == "NB_NODES: "){
            read = true;
            n_tree++;
            int n_nodes = std::stoi(line.substr(10, line.length()));
            for (int i=0;i<n_nodes; ++i){
                mini_nodes.push_back(Node(i));
            }
            nodes.push_back(mini_nodes);
        }
        else if (line.substr(0, 12) == "NB_CLASSES: "){
            n_classes = std::stoi(line.substr(12, line.length()));
        }

        else if (feat){
            if (line == ""){
                feat=false;
                this->features = features;
                continue;
            }
            std::vector<std::string> feat_str;
            std::istringstream ss(line);
            std::string field;
            while (std::getline(ss, field, ' ')) {
                feat_str.push_back(field);
            }
            std::string feat_name = feat_str[0];
            FeatureType feat_type = (feat_str[2] == "B") ? FeatureType::BINARY : FeatureType::NUMERICAL;
            features.push_back(Feature(feat_nbr,feat_type,feat_name));
            feat_nbr++;
        }
        else if (line.substr(0, 10) == "[FEATURES]"){
            feat = true;
        }
    }
    this->n_classes=n_classes;
    this->trees = trees;
    
}


Eigen::VectorXi Forest::distribution(std::vector<double> x){
    Eigen::VectorXi v(this->n_classes);
    v.fill(0);
    for (int i=0;i<this->trees.size(); i++){
        v[this->trees[i].klass(x)] += 1;
    }
    return v;
}


Eigen::MatrixXi Forest::full_distribution(DataFrame df){
    Eigen::MatrixXi m(df.data.size(),this->n_classes);
    for (int i=0; i < df.data.size(); i++){
        m.row(i) = this->distribution(df[i]);
    }
    return m;
}

Eigen::VectorXi Forest::klass(DataFrame df){
    Eigen::VectorXi argmax(df.data.size());
    Eigen::MatrixXi probs = this->full_distribution(df);
    int pos;
    for (int i = 0; i < probs.rows(); i++) {
        probs.row(i).maxCoeff(&pos);
        argmax(i) = pos;
    }
    return argmax;
}

/*int main(){
    DataFrame df = read_csv("/home/idrmal/RFC/resources/datasets/FICO/FICO.train1.csv");
    Forest rf("/home/idrmal/RFC/resources/forests/FICO/FICO.RF1.txt");
    std::cout<< "Nombre de classes : " << rf.n_classes <<std::endl;

    return 0;
}*/