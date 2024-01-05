/*
 * Essentially, std::ref is a function template in C++ that is used to create an
 *   std::reference_wrapper object that allows references to be passed where normally only 
 *   objects can be passed, such as with threads or bind expressions
 *
 * You are allowed to pass references in contexts where only copyable objects are expected
 *
 * One use case is with threads - since the std::thread constructor copies or moves the 
 *   arguments internally, you can use std::ref to pass a reference to a variable so that the
 *   thread can modify the external variable
 *
 * One other use case is with std::bind, when you want to bind a function argument to a 
 *   reference rather than a copy of an object
 */

#include <iostream>
#include <thread>
#include <functional>

void increment(int& x) {
    x++;
}

int main() {
    int value = 0;
    std::thread t {increment, std::ref(value)};
    t.join();

    std::cout << "Value after thread execution: " << value << std::endl;
}
