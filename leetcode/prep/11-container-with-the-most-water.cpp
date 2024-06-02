class Solution {
public:
    int maxArea(vector<int>& height) {
        int ans = 0;
        int lo = 0, hi = height.size() - 1;
        while (lo < hi) {
            ans = std::max(ans, (hi - lo) * std::min(height[lo], height[hi]));
            if (height[lo] < height[hi]) {
                lo++;
            } else {
                hi--;
            }
        }
        return ans;
    }
};
