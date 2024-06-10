class Solution {
public:
    int longestPalindromeSubseq(string s) {
        // dp[i][j] is the longest subsequence between i and j, inclusive
        int numChars = s.length();
        std::vector<std::vector<int>> dp (numChars, std::vector<int>(numChars, 1));
        
        for (int length = 2; length <= numChars; ++length) {
            for (int sIdx = 0; sIdx < numChars; ++sIdx) {
                int eIdx = sIdx + length - 1;
                if (eIdx >= numChars) {
                    continue;
                }
                
                int interval = 0;
                if (sIdx + 1 <= eIdx - 1) {
                    interval = dp[sIdx + 1][eIdx - 1];
                }

                dp[sIdx][eIdx] = std::max({
                    interval + (s[sIdx] == s[eIdx] ? 2 : 0),
                    dp[sIdx + 1][eIdx],
                    dp[sIdx][eIdx - 1]
                });
            }
        }

        return dp[0][numChars - 1];
    }
};
