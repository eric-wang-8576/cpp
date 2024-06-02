class Solution {
public:
    std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
        std::vector<std::vector<int>> ans;
        std::sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); ++i) {
            if (i == 0 || nums[i] != nums[i - 1]) {
                int j = i + 1;
                int k = nums.size() - 1;
                while (j < k) {
                    int sum = nums[i] + nums[j] + nums[k];
                    if (sum < 0) {
                        do {
                            ++j;
                        } while (j < nums.size() && nums[j] == nums[j - 1]);
                        
                    } else if (sum == 0) {
                        ans.push_back({nums[i], nums[j], nums[k]});
                        do {
                            ++j;
                        } while (j < nums.size() && nums[j] == nums[j - 1]);
                        
                        do {
                            --k;
                        } while (k > 0 && nums[k] == nums[k + 1]);
                        
                    } else {
                        do {
                            --k;
                        } while (k > 0 && nums[k] == nums[k + 1]);
                    }
                }
                
            }
        }

        return ans;
    }
};
