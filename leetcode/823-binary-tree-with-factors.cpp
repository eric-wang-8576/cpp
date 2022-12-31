class Solution {
public:
    int numFactoredBinaryTrees(vector<int>& arr) {
        long ans = 0;
        long modVal = 1'000'000'007;
        
        sort(arr.begin(), arr.end());
        unordered_map<int, long> numTrees;
        
        for (int i = 0; i < arr.size(); i++) {
            int currVal = arr[i];
            long currTrees = 1;
            for (int j = 0; j < i; j++) {
                if (arr[i] % arr[j] == 0) {
                    int quotient = arr[i]/arr[j];
                    if (numTrees.count(quotient)) {
                        currTrees = (currTrees + (numTrees.at(arr[j]) * numTrees.at(quotient))) % modVal;
                    }
                }
            }
            numTrees.insert({currVal, currTrees});
            ans = (ans + currTrees) % modVal;
        }
        
        return (int) (ans % modVal);
    }
};