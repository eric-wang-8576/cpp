/*
 * The constexpr specifier declares that it is possible to evaluate the value of the function
 *   or variable at compile time - such variables and functions can then be used where only
 *   compile time constant expressions are allowed
 *
 * A constexpr variable must be a literal-type, immediately initialized, and be a constant
 *   expression
 *
 * A constexpr function must not be virtual, a function-try-block, the return values must
 *   be LiteralType - it cannot contain
 *     - goto statements or labels
 *     - try-block
 *     - asm declaration
 *     - definition of a variable for which no initialization is performed
 *     - definition of a non-literal type variable
 *     - definition of a variable of static or thread storage duration 
 * 
 * The noexcept operator always returns true for a constant expression, it can be used to 
 *   check if a particular invocation of a constexpr function takes the constant expression
 *   branch 
 */

#include <iostream>

// C++11 constexpr function
constexpr int factorial(int n) {
    return n <= 1 ? 1 : (n * factorial(n - 1));
}

// C++14 constexpr function can use local variables and loops
constexpr int factorial__cxx14(int n) {
    int res = 1;
    while (n > 1) {
        res *= n;
        n--;
    }
    return res;
}

class conststr {
    const char* p ;
    std::size_t sz;
public:
    template<std::size_t N>
    constexpr conststr(const char(&a)[N]) : p(a), sz(N - 1) {}

    constexpr char operator[](std::size_t n) const {
        return n < sz ? p[n] : throw std::out_of_range("");
    }

    constexpr std::size_t size() const {
        return sz;
    }
};

constexpr std::size_t countlower(conststr s, std::size_t n = 0, std::size_t c = 0) {
    return n == s.size() ? c : 
        'a' <= s[n] && s[n] <= 'z' ? countlower(s, n + 1, c + 1)
                                   : countlower(s, n + 1, c);
}

constexpr std::size_t countlower2(conststr s, std::size_t n = 0, std::size_t c = 0) {
    int cnt = 0;
    for (int i = 0; i < s.size(); ++i) {
        if ('a' <= s[i] && s[i] <= 'z') {
            cnt++;
        }
    }
    return cnt;
}

template<int n>
struct constN {
    constN() {
        std::cout << n << std::endl;
    }
};

int main() {
    std::cout << "4! = ";
    constN<factorial(4)> out1;

    volatile int k = 8; // Disables optimization
//    constN<factorial(k)> out1; // Does not work 

    std::cout << k << "! = " << factorial(k) << std::endl; 

    std::cout << "The number of lowercase letters in \"Hello, world!\" is ";
    constN<countlower("Hello, world!")> cnt;

    std::cout << "The number of lowercase letters in \"Hello, world!\" is ";
    constN<countlower2("Hello, world!")> cnt2;

    constexpr int a[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    constexpr int length_a = sizeof(a) / sizeof(int);

    std::cout << "Array of length " << length_a << " has elements : ";
    for (int i = 0; i < length_a; ++i) {
        std::cout << a[i] << ' ';
    }
    std::cout << std::endl;
}
