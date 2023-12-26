/*
 * Avoid objects with names
 *
 * Avoid unnecessary std::move() - especially do not use it when returning a local object
 *
 * Constructors that initialize members from parameters, for which move operations are cheap, should
 *   take the argument by value and move it to the member
 *
 * Constructors that initialize members from parameters, for which move operations take a significant
 *   amount of time, should be overloaded for move semantics for best performance
 *
 * In general, creating and initializing new values from parameters, for which move operations are 
 *   cheap, should take the arguments by value and move - however do not take by value and move to 
 *   update/modify existing values
 *
 * Do not declare a virtual destructor in derived classes (unless you have to implement it)
 */

#include <iostream>

class Person {
public:
    std::string first, last;

    Person(const std::string& f, const std::string& l) : first{f}, last{l} {}

    Person(const std::string& f, std::string&& l) : first{f}, last{std::move(l)} {}

    Person(std::string&& f, const std::string& l) : first{std::move(f)}, last{l} {}

    Person(std::string&& f, std::string&& l) : first{std::move(f)}, last{std::move(l)} {}

    // For this setter, do not use move semantics
    void setFirstName(const std::string& s) {
        first = s;
    }
};
    
int main() {
    Person person {"E", "Wang"};
    person.setFirstName("Eric");
    std::cout << person.first << " " << person.last << std::endl;
}
