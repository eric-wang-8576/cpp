class Solution {
public:
    int minCostToSupplyWater(int n, vector<int>& wells, vector<vector<int>>& pipes) {
        using P = pair<int, int>;
        
        int ans = 0;
        vector<vector<P>> graph(n+1);
        priority_queue<P, vector<P>, greater<>> minHeap; // Stores (cost, vertex)
        
        for (const vector<int>& p: pipes) {
            const int house1 = p[0];
            const int house2 = p[1];
            const int cost = p[2];
            graph[house1].emplace_back(house2, cost);
            graph[house2].emplace_back(house1, cost);
        }
        
        for (int i = 1; i <= n; i++) {
            graph[0].emplace_back(i, wells[i - 1]);
            minHeap.emplace(wells[i - 1], i);
        }
        
        unordered_set<int> mst {{0}};
        
        while (mst.size() < n + 1) {
            const auto [cost, vertex] = minHeap.top();
            minHeap.pop();
            if (!mst.count(vertex)) {
                mst.insert(vertex);
                ans += cost;
                
                for (const auto [new_vertex, new_cost] : graph[vertex]) {
                    if (!mst.count(new_vertex)) {
                        minHeap.emplace(new_cost, new_vertex);
                    }
                }
            }
        }
        
        return ans;
    }
};