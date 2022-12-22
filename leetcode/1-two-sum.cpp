#include <unordered_map>

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> numToIndex;
        
        for (int i = 0; i < nums.size(); i++) {
            int currVal = nums[i];
            int diff = target - currVal;
            if (numToIndex.count(diff)) {
                return {numToIndex[diff], i};
            }
            numToIndex[currVal] = i;
        }   
        
        throw;
    }
};