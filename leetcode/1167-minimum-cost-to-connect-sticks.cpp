class Solution {
public:
    int connectSticks(vector<int>& sticks) {
        int ans = 0;
        
        priority_queue<int, vector<int>, greater<int>> pq;
        
        for (int i = 0; i < sticks.size(); i++) {
            pq.push(sticks[i]);
        }
        
        while (pq.size() > 1) {
            int first = pq.top();
            pq.pop();
            int second = pq.top();
            pq.pop();
            
            ans += first + second;
            pq.push(first + second);
        }
        
        return ans;
    }
};