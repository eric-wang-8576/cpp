#include "interval_tracker.hpp"

// The function std::map.upper_bound() returns an iterator to the first element in the map
//   whose key is greater than the specified key 

// lo and hi must be in range
void IntervalTracker::setRange(int lo, int hi, bool boolVal) {
    auto it = intervals.upper_bound(lo);

    // Check to see if we need to write in lo
    if (it != intervals.begin()) {
        auto prev = std::prev(it);
        if (prev->second != boolVal && lo != it->first) {
            intervals[lo] = boolVal;
        }
    } else {
        it->second = boolVal;
    }

    // Erase intermediate values and potentially an irrelevant value outside range
    hi++;
    boolVal = !boolVal;
    if (it->first == lo) {
        it++;
    }
    while (it != intervals.end()) {
        // If you are in the range, you are definitely overwritten
        if (it->first <= hi) {
            it = intervals.erase(it);
        } else {
            // If the next guy agrees with us, it's redundant
            if (it->second == boolVal) {
                intervals.erase(it);
            }
            break;
        }
    }

    // Write in range end
    if (hi <= maxVal) {
        intervals[hi] = boolVal;
    }
}

// val must be in range
bool IntervalTracker::contains(int val) {
    auto it = intervals.upper_bound(val);
    if (it != intervals.begin()) {
        auto prev = std::prev(it);
        return it->first == val ? it->second : prev->second;
    }
    return it->second;
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
