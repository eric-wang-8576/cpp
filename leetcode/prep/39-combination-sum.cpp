class Solution {
public:
    // Takes an index, and generates all possible combinations starting from that index
    // with that given target sum 

    void generateCombinations(
        std::vector<std::vector<int>>& ans,
        std::vector<int>& curr,
        std::vector<int>& candidates,
        int idx,
        int targetSum
    ) 
    {
        if (idx == candidates.size()) {
            return;
        }
        
        if (targetSum == 0) {
            ans.push_back(curr);
        } else if (targetSum - candidates[idx] >= 0) {
            curr.push_back(candidates[idx]);
            
            generateCombinations(ans, curr, candidates, idx, targetSum - candidates[idx]);

            curr.pop_back();
            
            generateCombinations(ans, curr, candidates, idx + 1, targetSum);
        } else {
            generateCombinations(ans, curr, candidates, idx + 1, targetSum);
        }
    }

    std::vector<std::vector<int>> combinationSum(std::vector<int>& candidates, int target) {
        std::vector<std::vector<int>> ans;
        std::vector<int> curr;
        generateCombinations(ans, curr, candidates, 0, target);
        return ans;
    }
};
