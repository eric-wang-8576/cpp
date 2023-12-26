/*
 * With noexcept, you can declare a conditional guarantee not to throw
 *
 * If you implement a move constructor, move assignment operator, or swap(), declare it with a 
 *   conditional noexcept expression
 *
 * For other functions, you might just want to mark them with an unconditional noexcept if they ever
 *   throw
 *
 * Destructors are always declared with noexcept (even when explicitly implemented)
 */

/*
 * The motivation for noexcept is that vector reallocation has the strong exception handling guarantee,
 *   which says that when an exception is thrown in the middle of the reallocation the vector is 
 *   guaranteed to roll back the vector to its previous state
 *
 * You cannot overload functions that only have different noexcept conditions
 *
 * In class hierarchies, a noexcept condition is part of the specified interface - overwriting a base
 *   class function that is noexcept in a derived class with a function that is not noexcept is an 
 *   error
 *
 * Every library function that the library working group can agree cannot throw and has a "wide 
 *   contract" (does not specify undefined behavior due to a precondition) should be marked as 
 *   unconditionally noexcept
 *
 * If a library swap function, move constructor, or move operation is "conditionally wide" (cannot be
 *   proven not to throw by applying the noexcept operator), then it should be marked as conditionally 
 *   noexcept 
 */

// static_assert(std::is_nothrow_move_constructible_v<Person>);

#include <iostream>
#include <type_traits>

class B {
    std::string s;
};

class Person {
private:
    std::string name;
public:
    Person(std::string str) {
        std::cout << "CONSTRUCT" << name << std::endl;
        name = str;
    }

    Person(Person&& p) 
        noexcept(std::is_nothrow_move_constructible_v<std::string> &&
                 noexcept(std::cout << name))
        : name{std::move(p.name)} 
     {
         std::cout << "MOVE ASSIGN" << std::endl;
     }
};

int main() {
    Person p {"Eric"};
    Person p2 = std::move(p);


    std::cout << std::boolalpha
              << std::is_nothrow_default_constructible<B>::value << std::endl
              << std::is_nothrow_copy_constructible<B>::value << std::endl
              << std::is_nothrow_move_constructible<B>::value << std::endl
              << std::is_nothrow_copy_assignable<B>::value << std::endl
              << std::is_nothrow_move_assignable<B>::value << std::endl;
}
