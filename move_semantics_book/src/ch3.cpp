/*
 * Move semantics is not passed through
 *
 * For every class, the move constructor and move assignment operator are automatically generated
 *
 * User-declaring a copy constructor, copy assignment operator, or destructor disables the automatic
 *   support of move semantics in a class - this does not impact the support in derived classes
 *
 * User-declaring a move constructor or move assignment operator disables the automatic support of
 *   copy semantics in a class - you get move-only types
 *
 * Never =delete a special move member function
 *
 * Do not declare a destructor if there is no specific need - there is no general need in classes
 *   derived from a polymorphic base class
 */

/*
 * The copy constructor is automatically generated when no move constructor is user-declared and no
 *   move assignemnt operator is user-declared
 *
 * The move constructor is automatically generated when no copy constructor is user-declared, no copy
 *   assignment operator is user-declared, no move assignment operator is user-declared, and no 
 *   destructor is user-declared
 *
 * The copy assignment operator is automatically generated when no move constructor is user-declared,
 *   and no move assignment operator is user-declared
 *
 * The move assignemnt operator is automatically generated when no copy constructor is user-declared,
 *   no move constructor is user-declared, no copy assignment operator is user-declared, and no 
 *   destructor is user-declared
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

class Customer {
private:
    std::string name;
    int val;
public:
    Customer(const std::string& n) : name{n} {
        std::cout << "CONSTRUCT " << name << std::endl;
        val = rand() % 100;
    }

    std::string getName() const {
        return name;
    }

    int getVal() const {
        return val;
    }

    friend std::ostream& operator<< (std::ostream& strm, const Customer& cust) {
        strm << '[' << cust.name << ": " << cust.val << ']';
        return strm;
    }

    // Copy constructor
    Customer(const Customer& cust) : name{cust.name}, val{cust.val} {
        std::cout << "COPY CONSTRUCT " << name << std::endl;
    }

    // Move Constructor
    Customer(Customer&& cust) noexcept : name{std::move(cust.name)}, val{std::move(cust.val)} {
        std::cout << "MOVE CONSTRUCT " << name << std::endl;
    }

    // Copy Assignment
    Customer& operator= (const Customer& cust) & {
        name = cust.name;
        val = cust.val;
        std::cout << "COPY ASSIGN " << name << std::endl;
        return *this;
    }

    // Move Assignment 
    Customer& operator= (Customer&& cust) & {
        name = std::move(cust.name);
        val = std::move(cust.val);
        std::cout << "MOVE ASSIGN " << name << std::endl;
        return *this;
    }
};


int main() {
    std::vector<Customer> coll;
    for (int i = 0; i < 8; ++i) {
        coll.push_back(Customer{"Test Customer " + std::to_string(i)});
    }

    std::sort(coll.begin(), coll.end(),
            [] (const Customer& c1, const Customer& c2) {
            return c1.getVal() < c2.getVal();
            });
}
