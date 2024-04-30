#include <thread>
#include <random>
#include <set>
#include <algorithm>
#include "cuckoo.hpp"

#define INITSIZE 4
#define MAXATTEMPTS 4

#define NACTIONS 100000
#define MAXVAL 100000

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(1, MAXVAL);

int genVal() {
    return dis(gen);
}

void print(std::set<int>& s) {
    std::vector<int> v;
    for (int val : s) {
        v.push_back(val);
    }

    std::sort(v.begin(), v.end());
    std::cout << "[";
    for (int i : v) {
        std::cout << i << ", ";
    }
    std::cout << "]";
}

int main() {
    Cuckoo c {INITSIZE, MAXATTEMPTS};
    std::set<int> s;

    for (int i = 0; i < NACTIONS; ++i) {
        if (i % 3 == 0) {
            // Add
            int val = genVal();
            bool cRes = c.add(val);
            bool sRes = s.insert(val).second;
            if (cRes != sRes) {
                std::cout << "Failure on add" << std::endl;
                print(s);
                return 0;
            }

        } else if (i % 3 == 1) {
            // Remove
            int val = genVal();
            bool cRes = c.remove(val);
            bool sRes = s.erase(val) == 1;
            if (cRes != sRes) {
                std::cout << "Failure on remove" << std::endl;
                print(s);
                return 0;
            }

        } else {
            // Contains
            int val = genVal();
            bool cRes = c.contains(val);
            bool sRes = s.find(val) != s.end();
            if (cRes != sRes) {
                std::cout << "Failure on contains" << std::endl;
                print(s);
                return 0;
            } 
        }
    }

    std::cout << "Passed!" << std::endl;
}
