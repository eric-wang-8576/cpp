class Solution {
public:
    int maxNumberOfApples(vector<int>& weight) {
        std::priority_queue<int, std::vector<int>, std::greater<int>> weights;
        for (int val : weight) {
            weights.push(val);
        }

        int total = 5000;
        int count = 0;
        while (weights.size() != 0 && total - weights.top() >= 0) {
            total -= weights.top();
            weights.pop();
            count++;
        }

        return count;
    }
};
