/*
 * For each class, clarify the state of moved-from objects - you have to ensure that they are at least
 *   destructrible - however, users of your class might expect/require more
 *
 * The requirements of functions of the C++ standard library also apply to moved-from objects
 *
 * Generated special move member functions might bring moved-from objects into a state such that a class
 *   invariant is broken - this might happen especially if 
 *     1. Classes have no default constructor with a determinate value (and no natural moved-from state)
 *     2. Values of members have restrictions (such as assertions)
 *     3. Values of members depend on each other
 *     4. Members with pointer-like semantics are used (such as smart pointers)
 * 
 * If the moved-from state breaks invariants or invalidates operations, you should fix this by using one
 *   of the following options
 *     1. Disable move semantics
 *     2. Fix the implementation of move semantics
 *     3. Deal with broken invariants inside the class and hide them to the outside 
 *     4. Relax the invariants of the class by documenting the constraints and preconditions for 
 *        moved-from objects
 */

/*
 * If the invariants of a class are broken, you have the following options
 *   1. Fix the move operations to bring the moved-from objects into a state that does not break
 *      the invariants
 *   2. Disable move semantics
 *   3. Relax the invariants that define all possible moved-from states also as valid - in particular,
 *      this might mean that member functions and functions that use the objects have to be implemented
 *      differently to deal with the new possible states 
 *   4. Document and provide a member function to check for the state of "broken invariants" so that the
 *      users of the type do not use an object of this type after it has been marked with std::move()
 */

#include <iostream>
#include <memory>
#include <string>

class SharedInt {
private:
    std::shared_ptr<int> sp;
    // Special value for moved-from objects
    inline static std::shared_ptr<int> movedFromValue{std::make_shared<int>(0)};

public:
    explicit SharedInt(int val) : sp{std::make_shared<int>(val)} {
        std::cout << "CONSTRUCT" << std::endl;
    }

    std::string asString() const {
        if (sp == movedFromValue) {
            return "NO VALUE";
        }
        return std::to_string(*sp);
    }

    SharedInt(SharedInt&& si) : sp{std::move(si.sp)} {
        std::cout << "COPY ASSIGN" << std::endl;
        si.sp = movedFromValue;
    }

    SharedInt& operator=(SharedInt&& si) noexcept {
        std::cout << "MOVE ASSIGN" << std::endl;
        if (this != &si) {
            sp = std::move(si.sp);
            si.sp = movedFromValue;
        }
        return *this;
    }

    SharedInt(const SharedInt&) =default;

    SharedInt& operator=(const SharedInt&) =default;
};

int main() {
    SharedInt si {3};
    SharedInt si2 {4};

    si2 = std::move(si);

    std::cout << si.asString() << std::endl;
    std::cout << si2.asString() << std::endl;
}
