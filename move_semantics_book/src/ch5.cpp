/*
 * You can overload member functions with different reference qualifiers
 *
 * Overload getters for expensive members with reference qualifiers to make them both safe and fast
 *
 * It can make sense to mark objects with std::move() even when calling member functions
 *
 * Use reference qualifiers in assignment operators
 */

#include <iostream>

class Person {
private:
    std::string name;

public:
    Person(std::string n) : name{std::move(n)} {}

    std::string getName() && {
        std::cout << "getName &&" << std::endl;
        return std::move(name);
    }

    const std::string& getName() const& {
        std::cout << "getName const&" << std::endl;
        return name;
    }
};

Person generatePerson() {
    Person person {"Wang"};
    return person;
}

int main() {
    std::vector<std::string> coll;
    Person person {"Eric"};

    std::cout << person.getName() << std::endl;
    std::cout << generatePerson().getName() << std::endl;
    std::cout << std::endl;

    coll.push_back(person.getName());
    coll.push_back(std::move(person).getName());
}
