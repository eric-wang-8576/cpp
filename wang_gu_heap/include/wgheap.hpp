#include <vector>
#include <iostream>

template<typename T>
class WangGuHeap {
    // Number of current elements contained in the heap
    // Also the next index to insert an element
    int n;

    // Heap representation is a complete binary tree where 
    //   children of index i are 2 * i + 1 and 2 * i + 2
    // Index 0 if present is the root of the heap
    // Resize policy - double when the arr reaches capacity
    std::vector<T> arr;

public:
    WangGuHeap() : n {0}, arr {0} {}

    // Inserts a value into the heap in O(log n) time
    void insert(T val);

    // Pops largest element in O(log n) time
    void pop();

    // Returns the largest element in O(1) time
    T top();

    // Prints heap for debugging
    void print();
};

template<typename T>
void WangGuHeap<T>::insert(T val) {
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

template<typename T>
void WangGuHeap<T>::pop() {
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

template<typename T>
T WangGuHeap<T>::top() {
    T ret = 0;
    if (n > 0) {
        ret = arr[0];
    }
    return ret;
}

template<typename T>
void WangGuHeap<T>::print() {
    std::cout << "Printing WangGuHeap: [ ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "]" << std::endl;
}

