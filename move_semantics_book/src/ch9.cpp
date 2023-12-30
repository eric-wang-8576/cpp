/*
 * Declarations with two ampersands (name&&) can be two different things:
 *   - If name is not a function template parameter is it an ordinary rvalue reference, binding only 
 *     to rvalues
 *   - If name is a function template parameter it is a universal reference, binding to all value 
 *     categories
 *
 * A universal/forwarding reference is a reference that can universally refer to all objects of any 
 *   type and value category - its type is
 *     - An lvalue reference (type&), if it binds to an lvalue
 *     - An rvalue reference (type&&), if it binds to an rvalue
 *
 * To perfectly forward a passed argument, declare the parameter as a universal reference of a template
 *   parameter of the function and use std::forward<>()
 *
 * The syntax std::forward<>() is a conditional std::move() - it expands to std::move() if its 
 *   parameter is bound to an rvalue
 *
 * It might make sense to mark objects with std::forward<>() even when calling member functions
 *
 * Universal references are the second-best option of all overload resolutions
 *
 * Avoid implementing generic constructors for one universal reference
 */

/*
 * C++11 introduced a special way to perfectly forward given arguments without any overloads but still
 *   keeping the type and the value category 
 *
 * An rvalue reference of a function template parameter is a universal/forwarding reference - they can
 *   bind to objects of all types and value categories
 *
 * The expression std::forward<T>(arg) essentially converts the expression to std::move(arg) if arg
 *   is an rvalue, or just arg if arg is an lvalue
 *
 * You can by default assume that after calling std::forward<>(x), x is in a valid but unspecified 
 *   state
 *
 * The universal reference is always the second-best option when selecting overloadingo
 *
 * Be careful when implementing a constructor with one single universal reference parameter
 */

#include <iostream>
#include <string>

template<typename T> 
void callFoo(T&& arg) {
    foo(std::forward<T>(arg));
}

template<typename T1, typename T2>
void callFoo(T1&& arg1, T2&& arg2) {
    foo(std::forward<T1>(arg1), std::forward<T2>(arg2));
}

template<typename... Ts>
void callFoo(Ts&&... args) {
    // ellipsis behind end of foward() since forward<> is called for all arguments
    foo(std::forward<Ts>(args)...); 
}

template<typename T>
void foo(T&& x) {
    x.print();
    std::forward<T>(x).getName();
}


class Person {
private:
    std::string name;
public:
    Person(std::string str) {
        name = std::move(str);
    }

    void print() const {
        std::cout << name << std::endl;
    }

    std::string getName() && {
        std::cout << "getName() &&"  << std::endl;
        return std::move(name);
    }

    const std::string& getName() & {
        std::cout << "getName() &" << std::endl;
        return name;
    }
};

int main() {
    Person p {"Eric"};

    callFoo(p);
    callFoo(Person{"Wang"});
}
