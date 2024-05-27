#include <iostream>

int computeModGPT(int base, int exp, int mod) {
    int res = 1;

    while (exp > 0) {
        if (exp & 1) {
            res = (res * base) % mod;
        }
        
        exp >>= 1;
        base = base * base % mod;
    }

    return res;
}

int computeMod(int base, int exp, int mod) {
    if (exp == 1) {
        return base % mod;

    } else if (exp & 1) {
        return base * computeMod(base, exp - 1, mod) % mod;

    } else {
        int rootMod = computeMod(base, exp / 2, mod);
        return rootMod * rootMod % mod;
    }
}

int main() {
    int base = 3;
    int exp = 100;
    int mod = 97;

    std::cout << computeModGPT(base, exp, mod) << std::endl;
    std::cout << computeMod(base, exp, mod) << std::endl;
}
