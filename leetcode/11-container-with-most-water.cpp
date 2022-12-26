class Solution {
public:
    int maxArea(vector<int>& height) {
        int ans = 0;
        
        int lo = 0;
        int hi = height.size() - 1;
        while (lo < hi) {
            int curr_size = min(height.at(lo), height.at(hi)) * (hi - lo);
            ans = max(ans, curr_size);
            if (height.at(lo) > height.at(hi)) {
                hi--;
            } else {
                lo++;
            }
        }
        
        return ans;
    }
};