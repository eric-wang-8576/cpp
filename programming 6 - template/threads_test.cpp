#include "thread_pool.hpp"

#include <iostream>

#define NUMTHREADS 32
#define NUMVALS 100

void printProduct(int a, int b) {
    std::cout << a * b << std::endl;
}

int main() {
    ThreadPool p {NUMTHREADS};
    for (int i = 0; i < NUMVALS; ++i) {
        p.enqueue(printProduct, i, i);
    }
}
