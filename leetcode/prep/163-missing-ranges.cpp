class Solution {
public:
    vector<vector<int>> findMissingRanges(vector<int>& nums, int lower, int upper) {
        std::vector<std::vector<int>> ans; // contains pairs
        nums.insert(nums.begin(), lower - 1);
        nums.push_back(upper + 1);
        for (int i = 0; i < nums.size(); ++i) {
            if (i > 0 && nums[i] - nums[i - 1] > 1) {
                ans.push_back({nums[i - 1] + 1, nums[i] - 1});
            }
        }
        return ans;
    }
};
