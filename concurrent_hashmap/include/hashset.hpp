#pragma once 
#include <iostream>
#include <set>
#include <mutex>

typedef std::set<int> bin;

class HashSet {
    std::vector<std::mutex> locks; // static sized array 
    std::vector<bin> bins; // contains bins, and expands based on capacity 

    int maxBinSize;

    int findLockIdx(int val);
    int findBinIdx(int val);
    void doubleCapacity();

public:
    HashSet(int numLocks, int maxBinSize) 
        : locks(numLocks), bins(numLocks), maxBinSize(maxBinSize) {}

    bool add(int val);
    bool remove(int val);
    bool contains(int val);
};
