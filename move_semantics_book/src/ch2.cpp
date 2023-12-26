/*
 * Rvalue references are declared with && and no const
 *
 * They can be initialized by temporary objects that do not have a name or non-const objects 
 *   marked with std::move()
 *
 * Rvalue references extend the lifetime of objects returned by value
 *
 * The function std::move() is a static_cast to the corresponding rvalue reference type - this
 *   allows us to pass a named object to an rvalue reference
 *
 * Objects marked with std::move() can also be passed to functions taking the argument by const
 *   lvalue reference but not taking a non-const lvalue reference
 *
 * Objects marked with std::move() can also be passed to functions taking the argument by value - 
 *   in that case, move semantics is used to initialize the parameter, which can make call-by-value
 *   pretty cheap
 *
 * Const rvalue references are possible but implementing them usually makes no sense
 *
 * Moved-from objects are in a valid but unspecified state - the C++ standard library guarantees that
 *   for its types - you can still use them providing you do not make any assumptions about their value
 */

#include <iostream>

/*
 * Takes the argument as const lvalue reference, is known as an in parameter
 * Fits a modifiable named object, a const named object, a temporary with no name, an object marked
 *   with std::move
 */
void foo(const std::string& arg) {}

/*
 * Takes the argument as non-const lvalue reference, known as an out parameter
 * Fits only a modifiable named object
 */
void foo(std::string& arg) {}

/*
 * Takes the argument as a non-const rvalue reference
 * Fits temporary objects with no name and non-const object marked with std::move()
 */
void foo(std::string&& arg) {}


int main() {

    // This is the same as foo(std::move(s))
    // foo(static_cast<std::string&&>(s)); 

    std::vector<std::string> allRows;
    std::string row;
    while (std::getline(std::cin, row)) {
        if (row == "exit") {
            std::cout << "Exiting!" << std::endl;
            return 0;
        } else if (row == "dump") {
            for (std::string& str : allRows) {
                std::cout << "\t" << str << std::endl;
            }
        }
        allRows.push_back(std::move(row));
    }
}
