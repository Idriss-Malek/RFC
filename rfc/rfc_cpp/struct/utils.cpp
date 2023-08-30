#include <iostream>
#include <vector>
#include <algorithm>

template<typename U>
struct IdentifiedObject {
    int id;
    U value;
};

template<typename U>
void idenumerate(std::vector<IdentifiedObject<U>>& it) {
    for (auto& x : it) {
        std::cout << x.id << ", " << x.value << std::endl;
    }
}

template<typename T>
T argmin(std::vector<T>& it, bool (*comp)(T, T) = nullptr) {
    T x = it[0];
    if (comp == nullptr) {
        comp = [](T a, T b) { return a < b; };
    }
    for (auto& i : it) {
        if (comp(i, x)) {
            x = i;
        }
    }
    return x;
}
