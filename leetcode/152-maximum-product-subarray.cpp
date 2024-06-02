class Solution {
public:
    int maxProduct(vector<int>& nums) {
        std::vector<int> negMax(nums.size(), 0);
        std::vector<int> posMax(nums.size(), 0);

        if (nums.size() == 1) {
            return nums[0];
        }
        
        // Fill in first value
        if (nums[0] > 0) {
            posMax[0] = nums[0];
        } else {
            negMax[0] = nums[0];
        }
        
        // Fill in rest of values
        for(int i = 1; i < nums.size(); ++i) {
            if (nums[i] > 0) {
                posMax[i] = std::max({ nums[i] * posMax[i - 1], nums[i] });
                negMax[i] = std::min({ nums[i] * negMax[i - 1], 0 });
            } else {
                posMax[i] = std::max({ nums[i] * negMax[i - 1], 0 });
                negMax[i] = std::min({ nums[i] * posMax[i - 1], nums[i] });
            }
        }
        
        return *std::max_element(posMax.begin(), posMax.end());
    }
};
