class Solution {
public:
    int trap(vector<int>& height) {
        // Contains pairs of (index, height), height cannot be 0
        // Monotonically decreasing stack 
        std::stack<std::pair<int, int>> stack;
        
        int total = 0;
        for (int i = 0; i < height.size(); ++i) {
            int currHeight = height[i];
            if (currHeight == 0) {
                continue;
            }
            
            int lastPoppedHeight = 0;
            // Calculate rainwater accumulation with smaller
            while (stack.size() != 0 && stack.top().second <= currHeight) {
                int prevHeight = stack.top().second;
                
                total += (prevHeight - lastPoppedHeight) * (i - stack.top().first - 1);
                
                stack.pop();
                lastPoppedHeight = prevHeight;
            }
            
            // Calculate rainwater accumulation against potential larger
            if (stack.size() != 0) {
                total += (currHeight - lastPoppedHeight) * (i - stack.top().first - 1);
            }
            stack.push(std::make_pair(i, currHeight));
        }

        return total;
    }
};
