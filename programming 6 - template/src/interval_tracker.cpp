#include "interval_tracker.hpp"

// The function std::map.upper_bound() returns an iterator to the first element in the map
//   whose key is greater than the specified key 

/*
 * Assumed that both lo and hi are in range
 */
void IntervalTracker::setRange(int lo, int hi, bool boolVal) {
    bool lastBoolVal; // value of the last flag that we overwrite

    auto it = intervals.upper_bound(lo);

    // Stage 1 - process interaction of lo with lower bound value if it exists
    if (it != intervals.begin()) {
        auto prev = std::prev(it); 
        lastBoolVal = prev->second;

        // we can ignore lo if the previous value is equivalent 
        if (prev->second != boolVal) {
            if (prev->first == lo) {
                // modify the flag if it is the first one, otherwise kill it
                if (prev == intervals.begin()) {
                    prev->second = boolVal;
                } else {
                    intervals.erase(prev);
                }
            } else {
                intervals[lo] = boolVal;
            }
        }
    } else {
        // we are at the first value
        lastBoolVal = boolVal; 
        it->second = boolVal;
    }

    // Stage 2 - delete all values in range
    while (it != intervals.end() && it->first <= hi) {
        lastBoolVal = it->second;
        it = intervals.erase(it);
    }

    // Stage 3 - process interaction of hi with subsequent values if they exist and if necessary
    if (lastBoolVal != boolVal) {
        if (it != intervals.end()) {
            // if the flag is right after the interval, by Invariant #2 we must be at boolVal
            if (it->first == hi + 1) {
                intervals.erase(it);
            } else {
                intervals[hi + 1] = lastBoolVal;
            }
        } else {
            // following Invariant #4
            if (hi + 1 <= maxVal) {
                intervals[hi + 1] = lastBoolVal;
            }
        }
    }
}

// val must be in range
bool IntervalTracker::contains(int val) {
    auto it = intervals.upper_bound(val);
    if (it != intervals.begin()) {
        auto prev = std::prev(it);
        return prev->second;
    }
    return intervals.end()->second;
}

void IntervalTracker::print() {
    std::cout << "IT -> ";
    for (auto& [val, boolVal] : intervals) {
        std::cout << val << ": " << (boolVal ? "true" : "false") << ", ";
    }
    std::cout << std::endl;
}

std::vector<bool> IntervalTracker::genBitVector() {
    std::vector<bool> bitvec(maxVal - minVal + 1);
    for (int i = minVal; i <= maxVal; ++i) {
        bitvec[i - minVal] = contains(i);
    }
    return bitvec;
}
