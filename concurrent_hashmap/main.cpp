#include <thread>
#include <random>
#include "hashset.hpp"

#define NTHREADS 100
#define NLOCKS 8
#define NVALS 10000

#define MAXBINSIZE 50
#define FAILURELIMIT 10000000

void addVals(HashSet& s, int tid) {
    int start = tid * NVALS;

    // Generate values and randomize order
    std::vector<int> vals;
    vals.reserve(NVALS);
    for (int i = 0; i < NVALS; ++i) {
        vals.push_back(start + i);
    }
    std::random_device rd;
    std::mt19937 g(rd());

    // Add values
    std::shuffle(vals.begin(), vals.end(), g);
    for (int i : vals) {
        s.add(i);
    }

    // Remove the odd values that were added 
    std::shuffle(vals.begin(), vals.end(), g);
    for (int i : vals) {
        if (i % 2) {
            s.remove(i);
        }
    }

    std::cout << "Thread #" + std::to_string(tid) + " finished." << std::endl;
}

// Verifies that all threads have successfully added and removed their values
void checkVals(HashSet& s) {
    static int failures = 0;
    bool success;
    while (true) {
        success = true;
        for (int val = 0; val < NTHREADS * NVALS; ++val) {
            bool contains = s.contains(val);
            bool even = val % 2 == 0;
            if ((contains && !even) || (!contains && even)) {
                success = false;
                break;
            }
        }
        if (success) {
            std::cout << "Success with " + std::to_string(failures) + 
                " prior failures." << std::endl;
            return;
        } else {
            failures++;
            if (failures > FAILURELIMIT) {
                std::cout << "Too many failures." << std::endl;
                return;
            }
        }
    }
}

int main() {
    HashSet s {NLOCKS, MAXBINSIZE};

    std::thread checker(checkVals, std::ref(s));
    
    std::vector<std::thread> threads;
    for (int tid = 0; tid < NTHREADS; ++tid) {
        threads.emplace_back(addVals, std::ref(s), tid);
    }

    for (auto& th : threads) {
        th.join();
    }
    checker.join();
}
