class Solution {
public:
    int subarraysDivByK(vector<int>& nums, int k) {
        // at each point, add the next number mod k, then store the number of occurrences so far
        int ans = 0;
        int* vals = new int[k]();
        
        int currSum = 0;
        vals[0] = 1;
        
        for (int i = 0; i < nums.size(); i++) {
            currSum = (currSum + (nums[i] % k) + k) % k;
            ans += vals[currSum];
            vals[currSum]++;
        }
        
        delete [] vals;

        return ans;
    }
};