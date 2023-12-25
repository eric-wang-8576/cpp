/*
 * The default constructor is called when objects of a class are declared with no arguments
 *
 * While rvalue references can be used for the type of any function parameter, they are seldom
 *   useful for uses other than the move constructor
 *
 * The keyword default defines a member function that would be implicitly defined if not deleted
 *
 * Classes that explictly define one copy/move constructor or one copy/move assignment but not both
 *   are encourgaed to specify either delete or default on the other special member functions they
 *   don't explicitly define
 *
 */

#include <iostream>
#include <string>

class String {
    std::string* ptr;
public:
    // Constructor
    String(const std::string& str) : ptr(new std::string(str)) {}

    // Destructor
    ~String() { delete ptr; }

    // Move Constructor
    String(String&& x) : ptr(x.ptr) { x.ptr = nullptr; }

    // Move Assignment
    String& operator= (String&& x) {
        delete ptr;
        ptr = x.ptr;
        x.ptr = nullptr;
        return *this;
    }

    const std::string& content() const {
        return *ptr;
    }

    String operator+(const String& rhs) {
        String str {""};
        str = this->content() + rhs.content();
        return str;
    }
};

int main() {
    String foo {"Str"};
    String bar = String {"ing"};

    foo = foo + bar;
    
    std::cout << "foo's content: " << foo.content() << std::endl;
}
