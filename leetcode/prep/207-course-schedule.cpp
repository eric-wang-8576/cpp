class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        std::vector<int> indegrees (numCourses, 0);
        // Maps each course to the courses that follow it
        std::vector<std::set<int>> graph (numCourses);
        
        // Construct the graph
        for (auto& edge : prerequisites) {
            int a = edge[0], b = edge[1];
            graph[b].insert(a);
            indegrees[a]++;
        }            

        // Push in initial courses
        std::vector<bool> canFinish (numCourses, 0);
        std::queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (indegrees[i] == 0) {
                q.push(i);
                canFinish[i] = true;
            }
        }

        // BFS
        while (q.size() != 0) {
            int course = q.front();
            q.pop();
            for (int nextCourse : graph[course]) {
                if (--indegrees[nextCourse] == 0) {
                    q.push(nextCourse);
                    canFinish[nextCourse] = true;
                }
            }
            
        }

        return std::all_of(canFinish.begin(), canFinish.end(),
            [] (bool val) {
                return val;
            }
        );
        
    }
};
