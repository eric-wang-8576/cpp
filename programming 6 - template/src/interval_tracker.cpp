#include "interval_tracker.hpp"

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

/*
 * Assumed that both lo and hi are in [minVal, maxVal] and lo <= hi
 */
void IntervalTracker::setRange(int lo, int hi, bool boolVal) {
    std::lock_guard<std::mutex> lock(mtx);

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
    std::lock_guard<std::mutex> lock(mtx);

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
