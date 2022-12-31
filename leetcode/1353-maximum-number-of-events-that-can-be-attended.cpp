class Solution {
public:
    int maxEvents(vector<vector<int>>& events) {
        int ans = 0;
        
        int currDay = 0;
        int currEvent = 0;
        int numEvents = events.size();
        
        priority_queue<int, vector<int>, greater<int>> pq; // Creates a min-heap
        sort(events.begin(), events.end()); 
        
        while (currEvent < numEvents || pq.size() > 0) {
            // If we have no events, then fast forward in time
            if (pq.size() == 0) {
                currDay = events[currEvent][0];
            }
                
            // Add events we can now attend
            while (currEvent < numEvents && events[currEvent][0] <= currDay) {
                pq.push(events[currEvent][1]);
                currEvent++;
            }

            // Attend the event 
            pq.pop();
            ans++;
            currDay++;  

            // Remove things we can no longer attend
            while (pq.size() > 0 && pq.top() < currDay) {
                pq.pop();
            }
        }
        
        return ans; 
    }
};