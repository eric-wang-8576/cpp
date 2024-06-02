class Solution {
public:
    void generate(
        std::vector<bool>& used,
        std::vector<int>& curr,
        std::vector<std::vector<int>>& ans,
        std::vector<int>& nums,
        int numUsed
    )
    {
        if (numUsed == nums.size()) {
            ans.push_back(curr);
        }
        for (int i = 0; i < nums.size(); ++i) {
            if (!used[i]) {
                curr.push_back(nums[i]);
                used[i] = true;
                
                generate(used, curr, ans, nums, numUsed + 1);

                curr.pop_back();
                used[i] = false;
            }
        }
    }
    
    std::vector<std::vector<int>> permute(std::vector<int>& nums) {
        std::vector<std::vector<int>> ans;
        std::vector<bool> used (nums.size(), false);
        std::vector<int> curr;
        generate(used, curr, ans, nums, 0);
        return ans;
    }
};
