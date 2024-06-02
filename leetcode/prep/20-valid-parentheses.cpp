class Solution {
public:
    bool isValid(string s) {
        std::stack<char> stack;
        std::map<char, char> m {{']', '['}, {')', '('}, {'}', '{'}};
        
        for (char c : s) {
            if (m.find(c) != m.end()) {
                if (stack.size() == 0 || stack.top() != m[c]) {
                    return false;
                }
                stack.pop();
            }  else {
                stack.push(c);
            }
        }
        
        return stack.size() == 0;
    }
};
