#include "wgheap.hpp"
#include <iostream>
#include <random>
#include <queue>

#define MAXVAL 1000000
#define NUMROUNDS 100
#define NUMVALS 100
#define DEBUG 0

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(1, MAXVAL);

int genVal() {
    return dis(gen);
}

int main() {
    // Create max heaps
    WangGuHeap<int> wgh;
    std::priority_queue<int, std::vector<int>, std::less<int>> pq;

    int heapSize;

    for (int i = 0; i < NUMROUNDS; ++i) {
        // Add NUMVALS values
        for (int i = 0; i < NUMVALS; ++i) {
            int val = genVal();
            wgh.insert(val);
            pq.push(val);
            if constexpr(DEBUG) {
                std::cout << "Inserting " << val << std::endl;
                wgh.print();
            }
            heapSize++;
        }

        // Remove half of them 
        for (int i = 0; i < NUMVALS / 2; ++i) {
            if (wgh.top() != pq.top()) {
                std::cout << "Error! " << wgh.top() << " " << pq.top() << std::endl;
                return 0;
            }
            if constexpr(DEBUG) {
                std::cout << "Popping " << wgh.top() << std::endl;
            }
            wgh.pop();
            pq.pop();
            if constexpr(DEBUG) {
                wgh.print();
            }
            heapSize--;
        }
    }

    // Empty the heap
    while (heapSize > 0) {
        if (wgh.top() != pq.top()) {
            std::cout << "Error! " << wgh.top() << " " << pq.top() << std::endl;
            wgh.print();
            return 0;
        }
        if constexpr(DEBUG) {
            std::cout << "Popping " << wgh.top() << std::endl;
        }
        wgh.pop();
        pq.pop();
        if constexpr(DEBUG) {
            wgh.print();
        }
        heapSize--;
    }

    std::cout << "Success!" << std::endl;
}
