#include "hashset.hpp"

int HashSet::findLockIdx(int val) {
    return val % locks.size();
}

// Must be called while bins.size() is static
int HashSet::findBinIdx(int val) {
    return val % bins.size();
}

void HashSet::doubleCapacity() {
    uint32_t oldSize = bins.size();

    for (auto& lock : locks) {
        lock.lock();
    }
    
    if (oldSize != bins.size()) {
        for (auto& lock : locks) {
            lock.unlock();
        }
        return;
    }

    // Create new bins array
    uint32_t newSize = oldSize * 2;
    std::vector<bin> newBins {newSize};
    for (auto& bin : bins) {
        for (int val : bin) {
            newBins[val % newSize].insert(val);
        }
    }

    // Move it into the object
    bins = std::move(newBins);

    for (auto& lock : locks) {
        lock.unlock();
    }

//    std::cout << "Resized from size " + std::to_string(oldSize) + 
//        " to size " + std::to_string(newSize) + "." << std::endl;
}

bool HashSet::add(int val) {
    int lockIdx = findLockIdx(val);

    locks[lockIdx].lock();
    int binIdx = findBinIdx(val);
    bool success = bins[binIdx].insert(val).second;
    int size = bins[binIdx].size();
    locks[lockIdx].unlock();

    if (size > maxBinSize) {
        doubleCapacity();
    }

    return success;
}

bool HashSet::remove(int val) {
    int lockIdx = findLockIdx(val);

    locks[lockIdx].lock();
    int binIdx = findBinIdx(val);
    bool success = bins[binIdx].erase(val);
    locks[lockIdx].unlock();

    return success;
}

bool HashSet::contains(int val) {
    int lockIdx = findLockIdx(val);

    locks[lockIdx].lock();
    int binIdx = findBinIdx(val);
    bool success = bins[binIdx].find(val) != bins[binIdx].end();
    locks[lockIdx].unlock();

    return success;
}
