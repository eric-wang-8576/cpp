/*
 * Move semantics allows us to optimize the copying of objects, where we no longer need the value -
 *   it can be used implicitly for unnamed temporary objects or local return values or explicitly with
 *   std::move()
 * 
 * Using std::move() means "I no longer need this value here" - it marks the object as movable - an 
 *   object marked with std::move is not destroyed
 * 
 * By declaring a function with a non-const rvalue reference, you define an interface where the 
 *   caller semantically claims it no longer needs the passed value - the implementer of the function
 *   can use this information to optimize its task by "stealing" the value or do any other 
 *   modification with the passed argument - the implementer also has to ensure that the passed 
 *   argument is in a valid state after the call
 *
 * Moved-from objects of the C++ standard library are still valid objects, but you no longer know
 *   their value
 *
 * Copy semantics is used as a fallback for move semantics - if there is no implementation taking 
 *   an rvalue reference, any implementation taking an ordinary const lvalue reference is used - this
 *   fallback is then used even if the object is explicitly marked with std::move()
 *
 * Calling std::move() for a const object usually has no effect
 *
 * If you return by value and not reference, do not declare the return value as a whole to be const
 *
 */

#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> createAndInsert() {
    std::vector<std::string> coll;
    coll.reserve(3);
    std::string s = "data";

    coll.push_back(s);
    coll.push_back(s + s);
    coll.push_back(std::move(s));

    return coll;
}

int main() {
    std::vector<std::string> v = createAndInsert();

    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << std::endl;
    }
}
