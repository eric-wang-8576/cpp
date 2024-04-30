#include "cuckoo.hpp"

int Cuckoo::hash1(int val) {
    return (val * 7) % size;
}

int Cuckoo::hash2(int val) {
    return (val * 13) % size;
}

bool Cuckoo::add(int val) {
    if (contains(val)) {
        return false;
    }

    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        // Attempt to insert into table 1, evict current value if necessary 
        int idx1 = hash1(val);
        if (t1[idx1] == -1) {
            t1[idx1] = val;
            numVals++;
            return true;
        }
        std::swap(val, t1[idx1]);

        // Attempt to insert into table 2, evict current value if necessary 
        int idx2 = hash2(val);
        if (t2[idx2] == -1) {
            t2[idx2] = val;
            numVals++;
            return true;
        }
        std::swap(val, t2[idx2]);
    }

    rehash();
    add(val);
    return true;
}

bool Cuckoo::remove(int val) {
    int idx1 = hash1(val);
    if (t1[idx1] == val) {
        t1[idx1] = -1;
        numVals--;
        return true;
    }

    int idx2 = hash2(val);
    if (t2[idx2] == val) {
        t2[idx2] = -1;
        numVals--;
        return true;
    }
    
    return false;
}

bool Cuckoo::contains(int val) {
    return t1[hash1(val)] == val || t2[hash2(val)] == val;
}

void Cuckoo::rehash() {
    int oldSize = size;
    size = oldSize * 2;
    std::vector<int> oldt1 = t1;
    std::vector<int> oldt2 = t2;

    // Clear and resize tables
    t1.resize(size);
    std::fill(t1.begin(), t1.end(), -1);
    t2.resize(size);
    std::fill(t2.begin(), t2.end(), -1);

    for (int val : oldt1) {
        if (val != -1) {
            add(val);
        }
    }

    for (int val : oldt2) {
        if (val != -1) {
            add(val);
        }
    }

    std::cout << "Resized table from " + 
        std::to_string(oldSize) + " to " + std::to_string(size) << std::endl;
}
