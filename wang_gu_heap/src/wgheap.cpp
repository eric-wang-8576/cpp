#include "wgheap.hpp"
#include <iostream>

void WangGuHeap::insert(int val) {
    // Double the vector if we don't have enough capacity
    int arrSize = arr.size();
    if (arrSize == n) {
        arr.resize(arrSize * 2);
    }

    // Add value
    int idx = n++;
    arr[idx] = val;

    // Propagate value upwards if necessary
    while (idx > 0) {
        int parentIdx = (idx - 1) / 2;
        if (arr[parentIdx] < arr[idx]) {
            std::swap(arr[parentIdx], arr[idx]);
            idx = parentIdx;
        } else {
            return;
        }
    }
}

void WangGuHeap::pop() {
    if (n > 0) {
        // Replace top element with last element
        arr[0] = arr[n - 1];
        n--;

        int idx = 0;
        // Propagate root downwards if necessary
        while (idx < n) {
            int leftIdx = 2 * idx + 1;
            int rightIdx = 2 * idx + 2;
            
            // Find the idx with the biggest value
            int swapIdx = idx;

            if (leftIdx < n && arr[leftIdx] > arr[swapIdx]) {
                swapIdx = leftIdx;
            }
            if (rightIdx < n && arr[rightIdx] > arr[swapIdx]) {
                swapIdx = rightIdx;
            }

            if (swapIdx == idx) {
                // Our heap invariants are satisfied
                return;
            } else {
                std::swap(arr[idx], arr[swapIdx]);
                idx = swapIdx;
            }
        }
    }
}

int WangGuHeap::top() {
    int ret = 0;
    if (n > 0) {
        ret = arr[0];
    }
    return ret;
}

void WangGuHeap::print() {
    std::cout << "Printing WangGuHeap: [ ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "]" << std::endl;
}

