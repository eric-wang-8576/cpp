class Solution {
public:
    bool canAttendMeetings(vector<vector<int>>& intervals) {
        std::sort(intervals.begin(), intervals.end(), 
            [] (vector<int>& i, vector<int>& j) {
                return i[0] < j[0];
            }
        );

        for (int i = 0; i < intervals.size(); ++i) {
            if (i > 0) {
                if (intervals[i][0] < intervals[i - 1][1]) {
                    return false;
                }
            }
        }
        return true;
        
    }
};
