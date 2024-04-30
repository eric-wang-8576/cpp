#pragma once
#include <iostream>

class Cuckoo {
    int size;
    std::vector<int> t1;
    std::vector<int> t2;

    int numVals;
    int hash1(int val);
    int hash2(int val);

    int maxAttempts;
    void rehash();

public:
    Cuckoo(int initSize, int numAttempts) 
        : size(initSize), t1(size, -1), t2(size, -1), maxAttempts(numAttempts) {}

    bool add(int val);
    bool remove(int val);
    bool contains(int val);
};


