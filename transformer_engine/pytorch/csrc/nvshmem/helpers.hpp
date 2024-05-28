#ifndef __TE_NVSHMEM_HELPERS_HPP__
#define __TE_NVSHMEM_HELPERS_HPP__

#include <vector>

template<typename T>
double error(std::vector<T> ref, std::vector<T> test) {
    double norm = 0;
    double diff = 0;
    ASSERT(ref.size() == test.size());
    for(size_t i = 0; i < ref.size(); i++) {
        norm += std::abs((double)ref[i]);
        diff += std::abs((double)ref[i] - (double)test[i]);
    }
    return diff / norm;
}

template<typename T>
std::vector<T> random(size_t count, int seed, int extra) {
    std::vector<T> out(count);
    for(size_t i = 0; i < count; i++) {
        out[i] = T(1 + (seed + i) % extra);
    }
    return out;
}

bool check(double value, double tolerance) {
    ASSERT(value >= 0);
    ASSERT(tolerance >= 0);
    if(value <= tolerance) {
        return true;
    } else {
        printf("FAILED, got %e <!= %e\n", value, tolerance);
        return false;
    }
}

#endif // __HELPERS_HPP__