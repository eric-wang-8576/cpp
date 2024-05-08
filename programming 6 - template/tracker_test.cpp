#include "interval_tracker.hpp"
#include <random>
#include <utility>

#define MINVAL 10
#define MAXVAL 25
#define NUMTRIALS 10000
#define DEBUG 1

int main() {
    IntervalTracker t {MINVAL, MAXVAL};

    static std::random_device rd;  // Obtain a random number from hardware
    static std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<> dis(MINVAL, MAXVAL); // Define the range


    std::vector<bool> bitvec(MAXVAL - MINVAL + 1, false);
    std::vector<bool> bitvec2(MAXVAL - MINVAL + 1, false);
    int first, second;
    for (int i = 0; i < NUMTRIALS; ++i) {
        first = dis(gen);
        second = dis(gen);
        if (first > second) {
            std::swap(first, second);
        }

        bool boolVal = i % 2 ? true : false;

        if constexpr(DEBUG) {
            std::cout << "Setting " << first << 
                " to " << second << " as " << (boolVal ? "true" : "false") << std::endl;
        }

        // Update both data structures
        for (int i = first; i <= second; ++i) {
            bitvec[i - MINVAL] = boolVal;
        }
        t.setRange(first, second, boolVal);
        bitvec2 = t.genBitVector();

        if constexpr(DEBUG) {
            std::cout << "[";
            for (int i = 0; i < MAXVAL - MINVAL + 1; ++i) {
                std::cout << bitvec[i] << ", ";
            }
            std::cout << "]" << std::endl;


            t.print();
            std::cout << "[";
            for (int i = 0; i < MAXVAL - MINVAL + 1; ++i) {
                std::cout << bitvec2[i] << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << std::endl;
        }

        // Compare
        if (bitvec != bitvec2) {

            std::cout << "[";
            for (int i = 0; i < MAXVAL - MINVAL + 1; ++i) {
                std::cout << bitvec[i] << ", ";
            }
            std::cout << "]" << std::endl;


            t.print();
            std::cout << "[";
            for (int i = 0; i < MAXVAL - MINVAL + 1; ++i) {
                std::cout << bitvec2[i] << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << std::endl;
            std::cout << "FAIL" << std::endl;
            return 0;
        }
    }

    std::cout << "Success" << std::endl;
    return 0;
}
