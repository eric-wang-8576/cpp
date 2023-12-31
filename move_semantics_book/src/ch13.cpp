/*
 * Move-only types allow us to move "owned" resources around without being able to copy them - the 
 *   copying special member functions are deleted
 *
 * You cannot use move-only types in std::initializer_lists
 *
 * You cannot iterate by value over collections of move-only types
 *
 * If you pass a move-only object to a sink function and want to ensure that you have lost ownership,
 *   explicitly release the resource directly afterwards
 */

/*
 * Move-only types simplify the management of unique resources, or types where objects represent a 
 *   value or own a resource for which copying does not make any sense 
 *
 * You cannot use std::initializer_lists for vectors of move-only types because they are usually passed
 *   by value, which requires copying of the elements 
 *
 * When passing move-only objects as arguments, use std::move() - if the parameter accepts by value, 
 *   then the function now owns the object - if the parameter accepts by reference, then the function
 *   may or may not take ownership of the object 
 *
 * When returning move-only objects by value, move semantics is automatically used - only when we have
 *   non-local data do we need an std::move() in the return statement 
 *
 * The C++ standard library uses different ways and names to check for a "moved-from" state that no 
 *   longer owns a resource 
 */

#include <iostream>

class MoveOnly {
private:
    std::string str;
public:
    MoveOnly(std::string p) : str(p) {}

    void print() {
        std::cout << "Str: " << str << std::endl;
    }

    // Copying Disabled
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator= (const MoveOnly&) = delete;

    // Moving Enabled
    MoveOnly(MoveOnly&& o) noexcept {
        str = std::move(o.str);
    }

    MoveOnly& operator= (MoveOnly&& o) noexcept {
        str = std::move(o.str);
        return *this;
    }
};

void sink(MoveOnly&& arg) {
    MoveOnly tmp = std::forward<decltype(arg)>(arg);
    tmp.print();
}

int main() {
    MoveOnly v {"v"};
    v.print();
    sink(std::move(v));
    v.print();
}
