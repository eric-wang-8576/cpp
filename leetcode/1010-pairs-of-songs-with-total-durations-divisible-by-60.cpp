class Solution {
public:
    int numPairsDivisibleBy60(vector<int>& time) {
        int ans = 0;
        int num[60] = {0};
        for (const int& duration : time) {
            ans += num[(60 - (duration % 60)) % 60];
            num[duration % 60]++;
        }
        return ans;
    }
};