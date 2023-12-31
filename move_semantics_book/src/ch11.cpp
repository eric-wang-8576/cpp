/*
 * Do not return by value with const (otherwise you disable move semantics for return values)
 *
 * Do not mark returned non-const rvalue references with const
 *
 * The code auto&& can be used to declare a universal reference that is not a parameter - just like 
 *   any universal reference, it can refer to all objects of any type and value category and its type
 *   is 
 *     - an lvalue reference (Type&) if it binds to an lvalue
 *     - an rvalue reference (Type&&) if it binds to an rvalue
 *
 * Use std::forward<decltype(ref)>(ref) to perfectly forward a universal reference ref that is declared
 *   with auto&&
 *
 * You can use universal references to refer to both const and non-const objects and use them multiple
 *   times without losing the information about their constness
 *
 * You can use universal references to bind to references that are proxy types
 *
 * Consider using auto&& in generic code that iterates over elements of a collection to modify them -
 *   that way, the code works for references that are proxy types 
 *
 * Use auto&& when declaring parameters of a lambda is a shortcut for declaring parameters that are
 *   universal references in a function template
 */

/*
 * If you pass a return value to another function directly, the value is passed perfectly - keeping 
 *   its type and value category 
 *
 * Do not mark values returned by value with const and do not mark returned non-const rvalue references
 *   with const 
 *
 * We want to declare range as a universal reference since we want to be able to bind it to every 
 *   range so that we can call it for begin() and end() without creating a copy or losing the 
 *   information about whether or not the range is const 
 *
 * If we deference an iterator for std::vector<bool>, we receive a std::vector<bool>::reference - this
 *   is because the implementation fo std::vector<bool> is a partial specialization of the 
 *   implementation of the primary template std::vector<T> where references to elements are objects
 *   of a proxy class that you can use like a reference - in this case, non-forwarding universal
 *   references allow us to bind to reference types that are not implemented as references, or bind
 *   to non-const objects provided as proxy types to manipulate objects
 *
 * Remember that lambdas are just an easy way to define functions objects (objects with operator()
 *   defined to allow their use as functions) - lambda definitions expand to compiler-defined classes
 *   or closure types
 */

#include <iostream>
#include <string>
#include <random>

void process(const std::string&) {
    std::cout << "const std::string&" << std::endl;
}

void process(std::string&) {
    std::cout << "std::string&" << std::endl;
}

void process(std::string&&) {
    std::cout << "std::string&&" << std::endl;
}

std::string& computeLRef(std::string& str) {
    return str;
}

std::string&& computeRRef(std::string&& str) {
    return std::move(str);
}

template<typename Coll, typename T>
void assign(Coll& coll, const T& value) {
    for (auto&& elem : coll) {
        elem = value;
    }
}

template<typename T>
void foo(T&& arg) {
    std::cout << arg << std::endl;
}

auto printString = [] (auto&& arg) {
    foo(std::forward<decltype(arg)>(arg));
};

int main() {
    std::string v {"v"};

    for (int i = 0; i < 10; ++i) {
        if (rand() % 2) {
            auto&& ret {computeLRef(v)};
            process(std::forward<decltype(ret)>(ret));
        } else {
            auto&& ret {computeRRef("Hi")};
            process(std::forward<decltype(ret)>(ret));
        }
    }

    std::vector<bool> coll {false, true, false};
    assign(coll, true);
    for (auto&& val : coll) {
        std::cout << val << std::endl;
    }
    
    const std::string c {"c"};
    printString(c);
}
