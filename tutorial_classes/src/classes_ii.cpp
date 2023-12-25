/*
 * Operators may be overloaded in two forms - either as a member function or a non-member function
 *   Operators overloaded as non-member functions take an object of the proper class as first argument
 *
 * They keyword this represents a pointer to the object whose member function is being executed
 *
 * Static members cannot be initialized inside the class, they must be iniitalized outside of it
 *   They can be referred to as a member of any object of that class or even directly by the class name
 *   Static functions cannot accesses non-static members of the class or use the keyword this
 *
 * When an object is declared as const, access to members is read-only and only const functions can
 *   be called - const functions cannot modify non-static data members - member functions can be
 *   overloaded on their constness 
 *
 * With a template specialization, we can create a different implementation for a template when
 *   a specific type is passed as template argument - we still have to precede this with 
 *   template <> because even though types are known, this is the specialization of a class template
 *   and thus requires to be noted as such 
 */

#include <iostream>

class CVector {
public:
    int x, y;
    CVector () {};
    CVector (int a, int b) : x(a), y(b) {}
    CVector operator+ (const CVector&);
    CVector& operator= (const CVector&);
};

CVector& CVector::operator= (const CVector& param) {
    x = param.x;
    y = param.y;
    return *this;
}

CVector CVector::operator+ (const CVector& param) {
    CVector temp;
    temp.x = x + param.x;
    temp.y = y + param.y;
    return temp;
}

CVector operator- (const CVector& lhs, const CVector& rhs) {
    CVector temp;
    temp.x = lhs.x - rhs.x;
    temp.y = lhs.y - rhs.y;
    return temp;
}

// MYPAIR

template <class T>
class mypair {
    T a, b;
public:
    mypair(T first, T second) {
        a = first;
        b = second;
    }
    T getMax();
};

template <class T>
T mypair<T>::getMax() {
    T retval;
    retval = a > b ? a : b;
    return retval;
}


int main() {
    CVector foo {3, 1};
    CVector bar {1, 2};

    CVector sum = foo + bar;
    CVector diff = foo - bar;
    
    std::cout << sum.x << "," << sum.y << std::endl;
    std::cout << diff.x << "," << diff.y << std::endl;

    mypair<int> obj {100, 75};
    std::cout << obj.getMax() << std::endl;

    return 0;
}
