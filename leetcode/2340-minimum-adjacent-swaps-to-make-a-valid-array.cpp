class Solution {
public:
    int minimumSwaps(vector<int>& nums) {
        int numVals = nums.size();
        
        int largest = 0;
        int smallest = 10001;
        for (int i = 0; i < numVals; i++) {
            if (nums[i] >= largest) {
                largest = nums[i];
            }
            if (nums[i] <= smallest) {
                smallest = nums[i];
            }
        }
        
        int smallestInd = 0;
        for (int i = 0; i < numVals; i++) {
            if (nums[i] == smallest) {
                smallestInd =i;
                break;
            }
        }
        
        int largestInd = numVals - 1;
        for (int i = numVals - 1; i >= 0; i--) {
            if (nums[i] == largest) {
                largestInd = i;
                break;
            }
        }

        int ans = (numVals - 1 - largestInd) + (smallestInd - 0);
        if (smallestInd > largestInd) {
            ans -= 1;
        }
        return ans;
    }
};