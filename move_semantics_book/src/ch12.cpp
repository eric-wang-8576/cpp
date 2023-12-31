/*
 * The code decltype(auto) is a placeholder type for deducing a value type from a value and a reference
 *   type from a reference
 *
 * Use decltype(auto) to perfectly return a value in a generic function
 *
 * In a return statement, never put parenthese around the return value/expression as a whole
 */

/*
 * The code decltype(auto) is a placeholder type that lets the compiler deduce the type at 
 *   initialization time 
 *
 * In contrast to auto&&, which is always a reference, decltype(auto) is sometimes just a value
 *
 * When using decltype(auto) as a return type, we use the rules of decltype as follows
 *   - if the expression returns/yields a plain value, then the value category is a prvalue, and
 *     decltype(auto) deduces a value type as "type"
 *   - if the expression returns/yields an lvalue reference, then the value category is an lvalue and
 *     decltype(auto) deduces an lvalue reference as "type&"
 *   - if the expression returns/yields an rvalue reference, then the value category is an xvalue and
 *     decltype(auto) deduces an rvalue reference as "type&&" 
 *
 * Never put additional parenthese around a returned name when using decltype(auto), because it will
 *   switch to the rules of expressions and always deduce an lvalue reference 
 *
 * With lambdas, we have to explicitly declare it with decltype(auto)
 */

#include <iostream>
#include<string>

#define CALL lambdaCall

template<typename Func, typename... Args>
decltype(auto) call(Func f, Args&&... args) {
    return f(std::forward<Args>(args)...);
}

template<typename Func, typename... Args>
decltype(auto) deferredCall(Func f, Args&&... args) {
    decltype(auto) ret {f(std::forward<Args>(args)...)};

    if constexpr(std::is_rvalue_reference_v<decltype(ret)>) {
        return std::move(ret);
    } else {
        return ret;
    }
}

auto lambdaCall = [] (auto f, auto&&... args) -> decltype(auto) {
    return f(std::forward<decltype(args)>(args)...);
};

std::string nextString() {
    return "Let's dance";
}

std::ostream& print(std::ostream& strm, const std::string& val) {
    return strm << "Value: " << val;
}

std::string&& returnArg(std::string&& arg) {
    return std::move(arg);
}

int main() {
    auto&& s = CALL(nextString); // returns temporary object 

    auto&& ref = CALL(returnArg, std::move(s)); // returns an rvalue reference to s 
    std::cout << "s: " << s << std::endl;
    std::cout << "ref: " << ref << std::endl;

    auto str = std::move(ref);
    std::cout << "s: " << s << std::endl;
    std::cout << "ref: " << ref << std::endl;
    std::cout << "str: " << str << std::endl;

    CALL(print, std::cout, str) << std::endl; // returns a reference to std::cout 
}
