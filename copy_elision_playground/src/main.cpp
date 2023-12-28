/*
 * Copy elision omits copy and move constructors, resulting in zero-copy pass-by-value semantics
 *
 * Copy elision is only effective when the object being initialized is known not to be a 
 *   potentially-overlaping subobject
 *
 * Compilers are permitted to omit the copy/move construction of class objects even if the copy
 *   constructor or destructor have observable side-effects - the objects are constructed
 *   directly into the storage where they would otherwise be copied/moved to
 *     1. In a return statement, the operand is a non-volatile object with automatic storage
 *        duration, and is the same class type as the function return type (NVRO)
 *     2. In the initialization of an object, when the source object is a nameless temporary
 *     3. In a throw-expression, when the operand is the name of a non-volatile object with
 *        automatic storage duration, which isn't a function parameter or catch clause parameter,
 *        and whose scope does not extend past the innermost try-block
 */

#include <iostream>

struct Noisy {
    Noisy() {
        std::cout << "CONSTRUCTOR " << this << std::endl;
    }

    Noisy(const Noisy&) {
        std::cout << "COPY CONSTRUCT " << this << std::endl;
    }

    Noisy(Noisy&&) {
        std::cout << "MOVE CONSTRUCT " << this << std::endl;
    }

    ~Noisy() {
        std::cout << "DESTRUCTOR " << this << std::endl;
    }
};

Noisy f() {
    Noisy v = Noisy();
    return v;
}

void g(Noisy arg) {
    std::cout << "&arg = " << &arg << std::endl;
}

int main() {
    Noisy v = f();

    std::cout << "&v = " << &v << std::endl;

    g(f());
}
