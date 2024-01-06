/*
 * The object std::pair constructs a pair object, which individually constructs the two component objects,
 *   with an initialization that depends on the constructor form involved:
 *     1. Default Constructor - constructs a pair with elements value-initialized
 *     2. Copy/Move Constructor - object is initialized with contents of the pr object, and the corresponding
 *        member is passed to the constructor of each of the members
 *     3. Initialization Constructor - member first is constructed with a and member second with b
 *     4. Piecewise Constructor - constructs the members first and second in-place, passing the elements of 
 *        first_args as arguments to the constructor of first, and the elements of second_args to the constructor
 *        of second 
 *
 * A std::pair is a specific case of an std::tuple with two elements 
 *
 * A std::tuple is an object capable to hold a collection of arguments - each element can be of a different type
 *
 * If std::is_trivially_destructible<Ti>::value is true for every Ti in Types, the destructor of tuple is trivial
 *
 * Since the size, types of elements, and ordering of elements are part of the tuple type signature, they must all 
 *   be available at compile time and can only depend on other compile-time information
 *
 * The object std::get(std::tuple t) returns a reference to the selected element of t - can specify a type
 *   instead of an integer, if the type is unique within the tuple
 *
 * The object std::tie creates a tuple of lvalue references to its arguments or instances of std::ignore
 */

#include <utility>
#include <string>
#include <iostream>
#include <tuple>
#include <set>
#include <cassert>

std::tuple<double, char, std::string> get_student(int id) {
    switch (id) {
        case 0: 
            return {3.8, 'A', "Lisa Simpson"};
        case 1:
            return {2.9, 'C', "Milhouse Van Houten"};
        case 2:
            return {1.7, 'D', "Ralph Wiggum"};
        case 3:
            return {0.6, 'F', "Bart Simpson"};
    }

    throw std::invalid_argument("id");
}


struct S {
    int n;
    std::string s;
    float d;

    friend bool operator<(const S& lhs, const S& rhs) noexcept {
        // This compares n, s, and d of lhs and rhs and returns first non-equal result
        // Returns false if all elements are equal
        // This comparison thus acts somewhat like a lexicographic comparison
        return std::tie(lhs.n, lhs.s, lhs.d) < std::tie(rhs.n, rhs.s, rhs.d);
    }
};

int main() {
    // Pair Simple Example
    std::pair<std::string, double> product1; // default initialization
    std::pair<std::string, double> product2 {"tomatoes", 2.30}; // value initialization
    std::pair<std::string, double> product3 {product2}; // copy constructor

    product1 = std::make_pair(std::string("lightbulbs"), 0.99);

    product2.first = "shoes";
    product2.second = 39.90;

    std::cout << "The price of " << product1.first << " is $" << product1.second << std::endl;
    std::cout << "The price of " << product2.first << " is $" << product2.second << std::endl;
    std::cout << "The price of " << product3.first << " is $" << product3.second << std::endl;

    // Tuple Simple Example
    std::tuple<int, char> foo {10, 'x'};
    auto bar = std::make_tuple("test", 3.1, 14, 'y');

    std::get<2>(bar) = 100; // returns a reference

    int myint;
    char mychar;

    std::tie(myint, mychar) = foo; // unpack elements
    std::tie(std::ignore, std::ignore, myint, mychar) = bar;

    mychar = std::get<3>(bar);

    std::get<0>(foo) = std::get<2>(bar);
    std::get<1>(foo) = mychar;

    std::cout << "foo contains: " << std::get<0>(foo) << " " << std::get<1>(foo) << std::endl;

    // Tuple Examples
    const auto student0 = get_student(0);
    std::cout << "ID: 0, "
              << "GPA: " << std::get<0>(student0) << ", "
              << "Grade: " << std::get<1>(student0) << ", "
              << "Name: " << std::get<2>(student0) << std::endl;


    const auto student1 = get_student(1);
    std::cout << "ID: 1, "
              << "GPA: " << std::get<double>(student1) << ", "
              << "Grade: " << std::get<char>(student1) << ", "
              << "Name: " << std::get<std::string>(student1) << std::endl;

    double gpa2;
    char grade2;
    std::string name2;
    std::tie(gpa2, grade2, name2) = get_student(2);
    std::cout << "ID: 2, "
              << "GPA: " << gpa2 << ", "
              << "Grade: " << grade2 << ", "
              << "Name: " << name2 << std::endl;
    
    const auto [gpa3, grade3, name3] = get_student(3);
    std::cout << "ID: 3, "
              << "GPA: " << gpa3 << ", "
              << "Grade: " << grade3 << ", "
              << "Name: " << name3 << std::endl;

    // Lexicographical Comparison
    std::set<S> set_of_s;
    S value {42, "Test", 3.14};
    std::set<S>::iterator iter;
    bool is_inserted;

    std::tie(iter, is_inserted) = set_of_s.insert(value);
    assert(is_inserted);

    auto position = [] (int w) {
        return std::tuple(1 * w, 2 * w);
    };

    auto [x, y] = position(1); // structured binding
    assert(x == 1 && y == 2);
    std::tie(x, y) = position(2); // std::tie
    assert(x == 2 && y == 4);

    std::tuple<char, short> coordinates {6, 9};
    std::tie(x, y) = coordinates; // implicit conversion
    assert(x = 6 && y == 9);
}
