class Solution {
public:
    int getNumDifferent(std::array<int, 26>& freq) {
        int sum = 0, max = 0;
        for (int val : freq) {
            sum += val;
            max = std::max(max, val);
        }
        return sum - max;
    }
    
    int characterReplacement(std::string s, int k) {
        int ans = 0;
        int lo = 0;
        std::array<int, 26> freq {0};
        for (int hi = 0; hi < s.length(); ++hi) {
            // Add new character
            freq[s[hi] - 'A']++;

            // Contract window to fulfill requirement
            while (getNumDifferent(freq) > k && lo < hi) {
                freq[s[lo++] - 'A']--;
            }

            ans = std::max(ans, hi - lo + 1);
        }
        
        return ans;
    }
};
