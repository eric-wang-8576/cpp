#include <map>
#include <iostream>
#include <algorithm>
#include <mutex>

class IntervalTracker {
    int minVal;
    int maxVal;
    std::mutex mtx;


    // Invariant #1
    // - If intervals[idx] = boolVal, then all values starting at idx and ending with the next largest key
    //   are boolVal
    //
    // Invariant #2
    // - If intervals[idx1] = boolVal, then the next largest key idx2 will have intervals[idx2] = !boolVal
    //
    // Invariant #3
    // - There will always be at least one interval, starting at value minVal and explicitly initialized
    // 
    // Invariant #4
    // - All indices will be in the range [minVal, maxVal]
    std::map<int, bool> intervals;

public:
    IntervalTracker(int minValP, int maxValP) : minVal(minValP), maxVal(maxValP) {
        intervals[minVal] = false;
    }

    IntervalTracker(const IntervalTracker&) = delete;
    IntervalTracker& operator=(const IntervalTracker&) = delete;
    IntervalTracker(IntervalTracker&&) = delete;
    IntervalTracker& operator=(IntervalTracker&&) = delete;

    void setRange(int lo, int hi, bool boolVal);
    bool contains(int val);
    void print();
    std::vector<bool> genBitVector();
};
