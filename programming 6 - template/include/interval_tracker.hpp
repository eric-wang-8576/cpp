#include <map>
#include <iostream>
#include <algorithm>

class IntervalTracker {
    const int minVal;
    const int maxVal;

    // If intervals[idx] = boolVal, then all values starting at idx and ending with the next key
    //   are boolVal
    // Invariant that ranges will always alternate
    // If empty, all values are by default false
    std::map<int, bool> intervals;

public:
    IntervalTracker(int minValP, int maxValP) : minVal(minValP), maxVal(maxValP) {
        intervals[minVal] = false;
    }

    void setRange(int lo, int hi, bool boolVal);
    bool contains(int val);
    void print();
    std::vector<bool> genBitVector();
};
