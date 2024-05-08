#include "interval_tracker.hpp"

// The function std::map.upper_bound() returns an iterator to the first element in the map
//   whose key is greater than the specified key 

// lo and hi must be in range
void IntervalTracker::setRange(int lo, int hi, bool boolVal) {
    bool lastBoolVal = intervals.begin()->second; // Value of the last flag we overwrite

    auto it = intervals.upper_bound(lo);

    // Check to see if we need to write in lo
    if (it != intervals.begin()) {
        auto prev = std::prev(it); // possible for prev to be equal to lo here
        if (prev->first == lo) {
            if (prev->second == boolVal) {
                // do nothing
                lastBoolVal = prev->second;
            } else {
                // kill the flag, move onto the next guy 
                if (prev == intervals.begin()) {
                    lastBoolVal = prev->second;
                    prev->second = boolVal;
                } else {
                    lastBoolVal = prev->second;
                    intervals.erase(prev);
                }
            }
        } else {
            if (prev->second == boolVal) {
                // do nothing
                lastBoolVal = prev->second;
            } else {
                // write new flag
                lastBoolVal = prev->second;
                intervals[lo] = boolVal;
            }
        }
    } else {
        lastBoolVal = boolVal; // check for accuracy
        it->second = boolVal;
    }

    // Erase intermediate flags and potentially an irrelevant flag outside range
    while (it != intervals.end()) {
        // If you are in the range, you are definitely overwritten
        if (it->first <= hi) {
            lastBoolVal = it->second;
            it = intervals.erase(it);
        } else {
            break;
        }
    }

    // Write in range end if the last guy disagreed with us and we aren't overwriting 
    if (lastBoolVal == boolVal) {
        // do nothing
    } else {
        // if the next space is occupied (by invariant must be boolVal, kill it
        // otherwise, write in the correct value
        if (it != intervals.end()) {
            if (it->first == hi + 1) {
                intervals.erase(it);
            } else {
                intervals[hi + 1] = lastBoolVal;
            }
        } else {
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
