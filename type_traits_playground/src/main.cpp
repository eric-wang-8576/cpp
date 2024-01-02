/*
 * A type trait is a simple template struct that contains a member constant, which 
 *   in turn holds the answer to the question that the type trait asks or the 
 *   transformation it performs
 *
 * For each of the traits, the compiler will generate a custom trait at compile 
 *   time that sets the value, type, or relevant member
 *
 * Intrinsics are special built-in functions provided by the compiler that give more
 *   insight about the type in question, thanks to the deep knowledge a compiler has
 *   on the program it takes in input 
 *
 * The <type_traits> header contains
 *   - Helper classes - standard classes to assist in creating compile-time constants
 *   - Type traits - classes to obtain characteristics of types in the form of compile-time
 *     constants
 *   - Type transformations - classes to obtain new types by applying specific 
 *     transformations to existing types
 *
 * Type traits include primary type categories, composite type categories, type properties,
 *   type features, type relationships, and property queries
 *
 * Type transformations include const-volatile qualifications, compound type alterations,
 *   and other type generators
 */

#include <iostream>

// Could contain a member called value or type
template<typename T>
struct example_type_trait;

void algo_signed(int i) {
    std::cout << "algo_signed" << std::endl;
}

void algo_unsigned(unsigned u) {
    std::cout << "algo_unsigned" << std::endl;
}

// Here, algo() acts as a dispatcher
template<typename T>
void algo(T t) {
    if constexpr(std::is_signed<T>::value) {
        algo_signed(t);
    } else if constexpr(std::is_unsigned<T>::value) {
        algo_unsigned(t);
    } else {
        static_assert("Must be signed or unsigned");
    }
}

int main() {
    algo(3);
    algo((unsigned) 3);
}
