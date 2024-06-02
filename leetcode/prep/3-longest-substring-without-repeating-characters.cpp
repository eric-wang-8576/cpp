class Solution {
public:
    int lengthOfLongestSubstring(std::string s) {
        std::unordered_map<char, int> freq;
        int ans = 0; 
        int lo = 0;
        for (int hi = 0; hi < s.length(); ++hi) {
            freq[s[hi]]++;
            while (freq[s[hi]] > 1 && lo < hi) {
                freq[s[lo++]]--;
            }
            ans = std::max(ans, hi - lo + 1);
        }
        return ans;
    }
};
