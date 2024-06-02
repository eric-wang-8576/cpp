class Solution {
public:
    void generate(
        int numPairs,
        int numOpen,
        int numClosed,
        std::string& curr,
        std::vector<std::string>& ans) 
    {
        if (numOpen == numPairs && numClosed == numPairs) {
            ans.push_back(curr);
            return;
        }

        // Try to add an open
        if (numOpen < numPairs) {
            curr.push_back('(');
            generate(numPairs, numOpen + 1, numClosed, curr, ans);
            curr.pop_back();
        }

        // Try to add a closed
        if (numClosed < numPairs && numClosed < numOpen) {
            curr.push_back(')');
            generate(numPairs, numOpen, numClosed + 1, curr, ans);
            curr.pop_back();
        }
    }
    
    std::vector<std::string> generateParenthesis(int n) {
        std::vector<std::string> ans;
        std::string curr;
        generate(n, 0, 0, curr, ans);
        return ans;
    }
};
