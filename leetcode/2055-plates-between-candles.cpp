class Solution {
public:
    vector<int> platesBetweenCandles(string s, vector<vector<int>>& queries) {
        // create list of candles
        // first idea is to binary search on the first plate after myself, then binary search first plate before myself, 
        vector<int> candles;
        
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '|') {
                candles.push_back(i);
            }
        }
        
        vector<int> ans;
        for (vector<int>& query : queries) {
            int toPush = 0;
            
            int left = query[0];
            int right = query[1];
            
            int lowerCandle = lower_bound(begin(candles), end(candles), left) - begin(candles);
            int upperCandle = upper_bound(begin(candles), end(candles), right) - begin(candles) - 1;
            if (lowerCandle < upperCandle) {
                int objectsInBetween = candles[upperCandle] - candles[lowerCandle] - 1;
                int candlesInBetween = upperCandle - lowerCandle - 1;
                toPush = objectsInBetween - candlesInBetween;
            }
            
            ans.push_back(toPush);
        }
        
        return ans;
    }
};