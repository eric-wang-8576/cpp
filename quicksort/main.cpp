#include <iostream>
#include <random>
#include <algorithm>
#include <utility>

#define NUMROUNDS 1000
#define NUMVALS 1000
#define MAXVAL 100

#define DEBUG 0


void dumpArray(const std::vector<int>& v, int n) {
    std::cout << "[ ";
    for (int i = 0; i < n; ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << "] " << std::endl;
}

// Returns a pivot index, with the guarantee that
//   all values to the right of the index are equal to or larger than all the values
//   before or equal to the index
int partition(std::vector<int>& v, int lo, int hi) {
    int pivot = v[lo];
    if constexpr(DEBUG) {
        std::cout << "Pivot Chosen: " << pivot << std::endl;
    }
    while (true) {
        // Increment lo until it is greater than or equal to pivot
        while (v[lo] < pivot) {
            lo++;
        }

        // Decrement hi until it is less than or equal to pivot
        while (v[hi] > pivot) {
            hi--;
        }

        if (lo < hi) {
            std::swap(v[lo++], v[hi--]);
            if constexpr(DEBUG) {
                dumpArray(v, NUMVALS);
            }
        } else {
            if constexpr(DEBUG) {
                std::cout << "Returning " << hi << std::endl;
            }
            return hi;
        }
    }
}


void quickSort(std::vector<int>& v, int lo, int hi) {
    if constexpr(DEBUG) {
        std::cout << "\nQuicksort Called with " << lo << ", " << hi << std::endl;
    }
    if (lo < hi) {
        int pivotIdx = partition(v, lo, hi);
        quickSort(v, lo, pivotIdx);
        quickSort(v, pivotIdx + 1, hi);
    }
}


std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(1, MAXVAL);

int genVal() {
    return dis(gen);
}

int main() {
    for (int i = 0; i < NUMROUNDS; ++i) {
        std::vector<int> v1;
        std::vector<int> v2;
        
        for (int i = 0; i < NUMVALS; ++i) {
            int val = genVal();
            v1.push_back(val);
            v2.push_back(val);
        }

        if constexpr(DEBUG) {
            dumpArray(v1, NUMVALS);
            std::cout << std::endl;
        }
        
        quickSort(v1, 0, NUMVALS - 1);
        std::sort(v2.begin(), v2.end());
        
        if (v1 != v2) {
            std::cout << "Failure!" << std::endl;
            dumpArray(v1, NUMVALS);
            dumpArray(v2, NUMVALS);
            return 0;
        }
    }

    std::cout << "Success!" << std::endl;
}
