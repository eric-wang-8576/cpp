class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        long ans = INT_MIN;
        long low = 0;
        long cumulative = 0;
        for (int val : nums) {
            cumulative += val;
            if (cumulative - low > ans)  {
                ans = cumulative - low;
            }
            if (cumulative < low) {
                low = cumulative;
            }
        }
        return int(ans);
    }
};
