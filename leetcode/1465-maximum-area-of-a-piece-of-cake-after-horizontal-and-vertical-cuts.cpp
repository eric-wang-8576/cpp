class Solution {
public:
    int maxArea(int h, int w, vector<int>& horizontalCuts, vector<int>& verticalCuts) {
        sort(horizontalCuts.begin(), horizontalCuts.end());
        sort(verticalCuts.begin(), verticalCuts.end());
        
        int height = h;
        int width = w;
        
        int maxWidth = 0;
        int prevCut = 0;
        for (int i = 0; i < verticalCuts.size(); i++) {
            maxWidth = max(maxWidth, verticalCuts[i] - prevCut);
            prevCut = verticalCuts[i];
        }
        maxWidth = max(maxWidth, width - prevCut);
        
        int maxHeight = 0;
        prevCut = 0;
        for (int i = 0; i < horizontalCuts.size(); i++) {
            maxHeight = max(maxHeight, horizontalCuts[i] - prevCut);
            prevCut = horizontalCuts[i];
        }
        maxHeight = max(maxHeight, height - prevCut);
        
        return (int) ((((long) maxWidth) * ((long) maxHeight)) % 1000000007);
    }
};