#include "../include/forest.hpp"
#include "../include/dataframe.hpp"

#include "../include/greedy.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <algorithm>
#include <ctime>        
#include <cstdlib>
#include <random> 
#include <thread>
#include <mutex>

std::mutex mtx;

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
    this->u = std::vector<int>(this->rf.trees.size(), 1);
}


void Greedy::compress(int iter, int tuple){
    std::vector<int>  index(this->rf.trees.size());
    std::iota(index.begin(), index.end(), 0);
    for (int it=0;it<iter;it++){
        std::vector<int> u(this->rf.trees.size(), 1);
        Eigen::MatrixXi probs = def_probs;
        Eigen::MatrixXi m1;
        Eigen::MatrixXi m2;
        Eigen::MatrixXi m3;

        std::srand ( unsigned ( std::time(0) ) );
        std::random_shuffle(index.begin(), index.end());

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

        if (tuple >= 2){
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
                            probs.noalias() += (m1+m2);
                        }
                    }
                }
            }
        }
        if (tuple==3){
            for (auto idx1: index) {
                for (auto idx2: index) {
                    for (auto idx3: index){
                        if (idx1 != idx2 && idx1 != idx3 && idx3 != idx2 && u[idx1] + u[idx2] + u[idx3] == 3){
                            m1 = rf.trees[idx1].full_distribution(df);
                            m2 = rf.trees[idx2].full_distribution(df);
                            m3 = rf.trees[idx3].full_distribution(df);
                            probs.noalias() -= (m1+m2+m3);
                            if (def_klass.isApprox(probs_to_class(probs), 0)){
                                u[idx1] = 0;
                                u[idx2] = 0;
                                u[idx3] = 0;
                            }
                            else{
                                probs.noalias() += (m1+m2+m3);
                            }
                        }
                    }
                    
                }
            }
        }
        
        
        if (std::accumulate(u.begin(), u.end(), 0) < std::accumulate(this->u.begin(), this->u.end(), 0)){
            std::lock_guard<std::mutex> lock(mtx);
            this->u = u;
        }
    }
}


int main(int argc, char* argv[]){
    std::string word = argv[1];
    int i = std::atoi(argv[2]);
    int tuple = std::atoi(argv[3]);
    std::stringstream ss;
    ss << "/home/idrmal/RFC/resources/forests/" << word<<"/"<<word<<".RF"<<i<<".txt";
    std::string rf_name = ss.str();
    Forest rf_(rf_name);
    ss.str("");
    ss << "/home/idrmal/RFC/resources/datasets/" << word<<"/"<<word<<".train"<<i<<".csv";
    std::string df_name = ss.str();
    DataFrame df_ = read_csv(df_name);
    Greedy greedy(df_, rf_);



    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int i=0;i<8; i++){
        threads.push_back(std::thread(&Greedy::compress, &greedy, 1, tuple));
    }
    for (std::thread& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;


    ofstream outfile("greedy_cpp_result.csv", ios::app);
    outfile << word<< " "<<i<<","<<greedy.u.size()<<","<<std::accumulate(greedy.u.begin(), greedy.u.end(), 0)<<","<<elapsed_time.count()<<" s \n";
    outfile.close();

    return 0;
}