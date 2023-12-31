/*
 * You can use universal references to bind to all const and non-const objects without losing the
 *   information about the constness of the object
 * 
 * You can use universal references to implement special handling for passed arguments with move 
 *   semantics without even using std::forward<>()
 *
 * To have a universal reference of a specific type, you need concepts/requirements or some template 
 *   tricks
 *
 * Only rvalue references of function template parameters are universal references - rvalue references
 *   of class template parameters, members of template parameters, and full specializations are 
 *   ordinary rvalue references, you can only bind rvalues
 *
 * When specifying the types of universal references explicitly, they act no longer as universal 
 *   references - use Type& to be able to pass lvalues then
 *
 * The C++ standards committee introduced forwarding reference as a "better" term for universal 
 *   reference - unfortunately, the term universal reference restricts the purpose of universal
 *   references to a common specific use case and creates the unnecessary confusion of having two terms
 *   for the same thing - therefore, use universal/forwarding reference to avoid even more confusion
 */

/*
 * A universal reference is the only way we can bind a reference to objects of any value category and
 *   still preserve whether or not it is const - the only other reference that binds to all objects,
 *   const&, loses the information about whether the passed argument is const or not
 */

#include <iostream>
#include <string>

void iterate(std::string::iterator begin, std::string::iterator end) {
    std::cout << "non-const stuff" << std::endl;
}

void iterate(std::string::const_iterator begin, std::string::const_iterator end) {
    std::cout << "const stuff" << std::endl;
}

template<typename T, 
         typename = typename std::enable_if<std::is_convertible<T, std::string>::value>::type>
void process(T&& coll) {
    iterate(coll.begin(), coll.end());
}

template<typename T>
void foo(T&& arg) {
    if constexpr(std::is_const_v<std::remove_reference_t<T>>) {
        std::cout << "is const!" << std::endl;
    } else {
        std::cout << "is not const!" << std::endl;
    }

    if constexpr(std::is_lvalue_reference_v<T>) {
        std::cout << "is lvalue!" << std::endl;
    } else {
        std::cout << "is not lvalue!" << std::endl;
    }
}

// Template specialization 

template<typename Coll, typename T>
void insert(Coll& coll, T&& arg) {
    std::cout << "primary template for universal reference of type T" << std::endl;
    coll.push_back(arg);
}

template<>
void insert(std::vector<std::string>& coll, const std::string& arg) {
    std::cout << "full specialization for type const std::string&" << std::endl;
    coll.push_back(arg);
}

// Template Parameter Deduction

template<typename T>
void insert(std::vector<std::remove_reference_t<T>>& vec, T&& elem) {
    std::cout << "template parameter deduction" << std::endl;
    vec.push_back(std::forward<T>(elem));
}

// Explicit Specification of Types for Universal References

template<typename T>
void f(T&& arg) {
    std::cout << arg << std::endl;
}


int main() {
    std::string v {"v"};
    const std::string c{"c"};

    process(v);
    process(c);
    process(std::string {"t"});
    process(std::move(v));
    process(std::move(c));

    foo(v);
    foo(c);

    f<std::string&>(v);

    std::vector<std::string> vec;
    insert(vec, v);
    std::cout << vec[0] << std::endl;
}
