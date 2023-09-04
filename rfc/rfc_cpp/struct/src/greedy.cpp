#include "../include/forest.hpp"
#include "../include/dataframe.hpp"

#include "../include/greedy.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <algorithm>
#include <ctime>        
#include <cstdlib>
#include <random> 



Eigen::VectorXi probs_to_class(Eigen::MatrixXi probs){
    Eigen::VectorXi argmax(probs.rows());
    int pos;
    auto f = [&pos](Eigen::VectorXi row) {
        row.maxCoeff(&pos);
        return pos;
    };
    for (int i; i<probs.rows(); i++){
        argmax(i) = f(probs.row(i));
    }
    return argmax;
}

Greedy::Greedy(DataFrame df_, Forest forest_): rf(forest_), df(df_) { // Use member initialization list
    this->def_probs = this->rf.full_distribution(this->df);
    this->def_klass = probs_to_class(this->def_probs);
    std::cout<<rf.trees.size()<<std::endl;
    this->u = std::vector<int>(this->rf.trees.size(), 1);
}


void Greedy::compress(int iter){
    std::vector<int>  index(this->rf.trees.size());
    std::iota(index.begin(), index.end(), 0);
    for (int it=0;it<iter;it++){
        std::vector<int> u(this->rf.trees.size(), 1);
        Eigen::MatrixXi probs = def_probs;
        Eigen::MatrixXi m1;
        Eigen::MatrixXi m2;

        std::srand ( unsigned ( std::time(0) ) );
        std::random_shuffle(index.begin(), index.end());

        for (auto idx1: index) {
            for (auto idx2: index) {
                if (idx1 != idx2 && u[idx1] + u[idx2] == 2){
                    m1 = rf.trees[idx1].full_distribution(df);
                    m2 = rf.trees[idx2].full_distribution(df);
                    probs.noalias() -= (m1+m2);
                    if (def_klass.isApprox(probs_to_class(probs), 0)){
                        u[idx1] = 0;
                        u[idx2] = 0;
                    }
                    else{
                        probs+= (m1+m2);
                    }
                }
            }
        }
        for (auto idx1: index) {
            if (u[idx1] == 1){
                m1 = rf.trees[idx1].full_distribution(df);
                probs.noalias() -= m1;
                if (def_klass.isApprox(probs_to_class(probs), 0)){
                    u[idx1] = 0;
                }
                else{
                    probs.noalias() += m1;
                }
            }
        }
        if (std::accumulate(u.begin(), u.end(), 0) < std::accumulate(this->u.begin(), this->u.end(), 0)){
            this->u = u;
        }
    }
}


int main(){
    Forest rf_("/home/idrmal/RFC/resources/forests/FICO/FICO.RF1.txt");
    std::cout<<"Forest Created"<<std::endl;
    DataFrame df_ = read_csv("/home/idrmal/RFC/resources/datasets/FICO/FICO.train1.csv");
    std::cout<<"DataFrame Created"<<std::endl;
    Greedy greedy(df_, rf_);
    std::cout<<"Greedy Created"<<greedy.u[5]<<std::endl;


    std::cout<<"Original size : "<<std::accumulate(greedy.u.begin(), greedy.u.end(), 0)<<std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    greedy.compress(10);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;

    std::cout<<"Compressed size : "<<std::accumulate(greedy.u.begin(), greedy.u.end(), 0)<< " | Time elapsed : "<< elapsed_time.count() << " s"<<std::endl;
    
    

    return 0;
}