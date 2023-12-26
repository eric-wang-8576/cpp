/*
 * Any expression in a C++ program belongs to one of three primary value categories
 *   1. Locator value, or lvalue is a named object or string literal
 *   2. Pure rvalue, or prvalue is an unnamed temporary object
 *   3. Expiring value, or xvalue is an object marked with std::move()
 *
 * Whether a call or operation in C++ is valid depends on both the type and the value category
 *
 * Rvalue references of types can only bind to prvalues or xvalues
 *
 * Implicit operations might change the value category of a passed argument
 *
 * Passing an rvalue to an rvalue reference binds it to an lvalue
 *
 * Move semantics is not passed through
 *
 * Functions and references to functions are always lvalues
 *
 * For rvalues, plain value members have move semantics but reference or static members do not
 *
 * The keyword decltype can either check for the declared type of a passed name or for the type and 
 *   the value category of a passed expression
 */

/*
 * Examples of lvalues include
 *   - An expression that is just the name of a variable, function, or data member
 *   - An expression that is just a string literal
 *   - The return value of a function if it is declared to return an lvalue reference
 *   - Any reference to a function, even if it is marked with std::move()
 *   - The result of the built-in unary * operator
 *
 * Examples of prvalues include
 *   - Expressions that consist of a built-in literal that is not a string literal
 *   - The return type of a function if it is declared to return by value
 *   - The result of the built-in unary operator
 *   - A lambda expression
 *
 * Examples of xvalues include
 *   - The result of marking an object with std::move()
 *   - A cast to an rvalue reference of an object type
 *   - The returned value of a function if it is declared to return an rvalue reference
 *   - A non-static value member of an rvalue
 *
 * Typically, glvalues refer to generalized lvalues or expressions for locations of long-living 
 *   objects or functions
 *
 * On the other hand, prvalues are expressions for short-living values for initializations
 *
 * If we pass a prvalue to a parameter expecting a glvalue, it is converted to an xvalue
 *
 * While reference and static members of rvalues are lvalues, plain data members of rvalues are xvalues
 *
 * Usually, you should take an argument either by value or reference (with as many reference overloads
 *   as you think are useful) but never both
 *
 * When checking the value category of an expression, 
 *   - For a prvalue it just yields the type: type
 *   - For an lvalue it yields its type as an lvalue reference: type&
 *   - For an xvalue it yields its type as an xvalue reference: type&&
 */

#include <iostream>

void rvFunc(std::string&& str) {
    std::cout << std::boolalpha 
              << "Is name type same as std::string: " 
              << std::is_same<decltype(str), std::string>::value 
              << std::endl
              << "Is name type same as std::string&: " 
              << std::is_same<decltype(str), std::string&>::value 
              << std::endl
              << "Is name type same as std::string&&: "
              << std::is_same<decltype(str), std::string&&>::value
              << std::endl
              << "Is name type a reference: "
              << std::is_reference<decltype(str)>::value
              << std::endl
              << "Is name type a lvalue reference: "
              << std::is_lvalue_reference<decltype(str)>::value
              << std::endl
              << "Is name type a rvalue reference: "
              << std::is_rvalue_reference<decltype(str)>::value
              << std::endl

              << "Is expression type same as std::string: " 
              << std::is_same<decltype((str)), std::string>::value 
              << std::endl
              << "Is expression type same as std::string&: " 
              << std::is_same<decltype((str)), std::string&>::value 
              << std::endl
              << "Is expression type same as std::string&&: "
              << std::is_same<decltype((str)), std::string&&>::value
              << std::endl
              << "Is expression type a reference: "
              << std::is_reference<decltype((str))>::value
              << std::endl
              << "Is expression type a lvalue reference: "
              << std::is_lvalue_reference<decltype((str))>::value
              << std::endl
              << "Is expression type a rvalue reference: "
              << std::is_rvalue_reference<decltype((str))>::value
              << std::endl;
}

int main() {
    rvFunc("Eric");
}
