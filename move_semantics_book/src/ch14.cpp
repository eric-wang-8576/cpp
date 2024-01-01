/*
 * The C++ standard library provides special algorithms to move multiple elements, called std::move() 
 *   and std::move_backwards() - they leave elements in a moved-from state 
 *
 * Removing algorithms can leave elements in a moved-from state
 *
 * Move iterators allow us to use move semantics when iterating over elements - you can use these
 *   iterators in algorithms or other places such as constructors where ranges are used to initialize
 *   or set values - however, ensure that the iterators use each element only once 
 */

/*
 * Removing algorithms do not really remove elements - they only move the values of all the elements
 *   that are not removed to the front of the processed range and return the new end
 *
 * The functions std::remove(), std::remove_if(), and std::unique() can all leave elements in a 
 *   moved-from state
 *
 * By using move iterators, you can use move semantics even in other algorithms and in general 
 *   wherever input ranges are taken
 *
 * For algorithms with callables, allowing the specification the detailed functionality, the elements
 *   are passed to the callable with std::move() - inside the callable, you can decide how to deal 
 *   with them
 *     - take the argument by value to always move/steal the value or resource
 *     - take the argument by rvalue/universal reference to decide which value/resource to move/steal
 *
 * The helper function std::make_move_iterator() is used so that you do not have to specify the element
 *   type when declaring the iterator 
 *
 * We can move all elements from a list to a vector with std::make_move_iterator()
 */

#include <iostream>
#include <cassert>
#include <string>
#include <algorithm>
#include <list>

class Email {
private:
    std::string value;
    bool movedFrom {false};
public:
    Email(const std::string& val) : value(val) {
        assert(value.find('@') != std::string::npos);
    }

    Email(const char* val) : Email{std::string(val)} {} // Enable implicit conversions

    std::string getValue() const {
        assert(!movedFrom);
        return value;
    }

    Email(Email&& e) noexcept : value{std::move(e.value)}, movedFrom{e.movedFrom} {
        e.movedFrom = true;
    }

    Email& operator=(Email&& e) noexcept {
        value = std::move(e.value);
        movedFrom = e.movedFrom;
        e.movedFrom = true;
        return *this;
    }

    Email(const Email&) = default;
    Email& operator=(const Email&) = default;

    friend std::ostream& operator << (std::ostream& strm, const Email& e) {
        return strm << (e.movedFrom ? "MOVED-FROM" : e.value);
    }
};

template<typename T>
void print(const std::string& name, const T& coll) {
    std::cout << name << " (" << coll.size() << " elements): ";
    for (const auto& elem : coll) {
        std::cout << " \"" << elem << "\"";
    }
    std::cout << std::endl;
}

// Gets moved values from rvalues
void process(std::string s) {
    std::cout << "- process(" << s << ")" << std::endl;
}

int main() {
    // Email
    std::vector<Email> coll {"tom@domain.de", 
                             "jill@company.com", 
                             "sarah@domain.de", 
                             "hana@company.com"};
 
    std::cout << "before all elements:" << std::endl;
    for (const auto& elem : coll) {
        std::cout << " '" << elem << "'\n";
    }

    auto newEnd = std::remove_if(coll.begin(), coll.end(), 
                                 [] (const Email& e) {
                                    auto&& val = e.getValue();
                                    return val.size() > 2 && val.substr(val.size() - 3) == ".de";
                                 });

    std::cout << "remaining elements:" << std::endl;
    for (auto pos = coll.begin(); pos != newEnd; ++pos) {
        std::cout << " '" << *pos << "'\n";
    }

    std::cout << "after all elements:" << std::endl;
    for (const auto& elem : coll) {
        std::cout << " '" << elem << "'\n";
    }

    // Move iterators
    std::vector<std::string> strs {"aaa", "aaaa", "aaaaa", "bbb", "bbbb", "bbbbb"};
    print("strs", strs);

    std::for_each(strs.begin(),
                  strs.end(),
                  [] (auto& elem) {
                    if (elem.size() == 3) {
                        process(std::move(elem));
                    }
                });
    print("strs", strs);


    std::for_each(std::make_move_iterator(strs.begin()),
                  std::make_move_iterator(strs.end()),
                  [] (auto&& elem) {
                    if (elem.size() == 4) {
                        process(std::move(elem));
                    }
                });
    print("strs", strs);
    
    // Move iterators between list and string

    std::list<std::string> src {"I", "am", "Eric", "Wang"};
    std::vector<std::string> dest {std::make_move_iterator(src.begin()), 
                                   std::make_move_iterator(src.end())};

    print("src", src);
    print("dest", dest);
}
