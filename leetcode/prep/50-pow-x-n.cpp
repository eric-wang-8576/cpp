class Solution {
public:
    double exp(double x, long long n) {
        if (n < 0) {
            return 1.0 / exp(x, -n);
            
        } else if (n == 0) {
            return 1;
            
        } else {
            if (n & 1) {
                return exp(x, n - 1) * x;
            } else {
                double val = exp(x, n / 2);
                return val * val;
            }
        }
    }
    
    double myPow(double x, int n) {
        return exp(x, n);
    }
};
