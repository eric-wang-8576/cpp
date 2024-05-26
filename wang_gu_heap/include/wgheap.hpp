#include <vector>

class WangGuHeap {
    // Number of current elements contained in the heap
    // Also the next index to insert an element
    int n;

    // Heap representation is a complete binary tree where 
    //   children of index i are 2 * i + 1 and 2 * i + 2
    // Index 0 if present is the root of the heap
    // Resize policy - double when the arr reaches capacity
    std::vector<int> arr;

public:
    WangGuHeap() : n {0}, arr {0} {}

    // Inserts a value into the heap in O(log n) time
    void insert(int val);

    // Pops largest element in O(log n) time
    void pop();

    // Returns the largest element in O(1) time
    int top();

    // Prints heap for debugging
    void print();
};
