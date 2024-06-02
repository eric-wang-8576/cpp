class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int ans = 0;
        int cumulative = 0;
        
        std::map<int, int> freq; // maps cumulative sums to frequencies seen
        freq[0]++;
        
        for (int val : nums) {
            cumulative += val;
            if (freq.find(cumulative - k) != freq.end()) {
                ans += freq[cumulative - k];
            }
            freq[cumulative]++;
        }
        
        return ans;
    }
};
