class Solution {
public:
    string removeDuplicates(string s, int k) {
        int slow = 0;
        stack<int> freq;
        
        for (int fast = 0; fast < s.size(); ++slow, ++fast) {
            s[slow] = s[fast];
            if (slow == 0 || s[slow] != s[slow - 1]) {
                freq.push(1);
            } else {
                if (++freq.top() == k) {
                    slow -= k;
                    freq.pop();
                }
            }
        }
        return s.substr(0, slow); 
    }
};