class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        // 1 is land, 0 is water, 2 is visited
        std::vector<std::vector<int>> dirs {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        
        int numIslands = 0, numRows = grid.size(), numCols = grid[0].size();
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                if (grid[r][c] == '1') {
                    numIslands++;

                    // BFS from this location
                    std::queue<std::pair<int, int>> q;
                    q.push(std::make_pair(r, c));
                    while (q.size() != 0) {
                        std::pair<int, int> coords = q.front();
                        q.pop();
                        for (auto dir : dirs) {
                            int newR = coords.first + dir[0];
                            int newC = coords.second + dir[1];
                            if (0 <= newR && newR < numRows
                                && 0 <= newC && newC < numCols
                                && grid[newR][newC] == '1'
                            ) {
                                grid[newR][newC] = '0';
                                q.push(std::make_pair(newR, newC));
                            }
                        }
                    }
                    
                }
            }
        }

        return numIslands;
        
    }
};
