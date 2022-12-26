class Solution {
public:
    vector<int> killProcess(vector<int>& pid, vector<int>& ppid, int kill) {
        unordered_map<int, vector<int>> parentToChild;
        
        for (int i = 0; i < pid.size(); i++) {
            const int child = pid[i];
            const int parent = ppid[i];
            
            parentToChild[parent].push_back(child);
        }
        
        vector<int> ans;
        unordered_set<int> seen;
        queue<int> q;
        
        q.push(kill);
        while (q.size() != 0) {
            int curr = q.front();
            q.pop();
            ans.push_back(curr);
            for (const auto child: parentToChild[curr]) {
                if (seen.find(child) == seen.end()) {
                    seen.insert(child);
                    q.push(child);
                }
            }
        }
        
        return ans;
    }
};