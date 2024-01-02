/*
 * The feature std::optional provides a standardized way to represent optional values, eliminating
 *   the need for null pointers and enhancing code readability 
 *
 * By design, std::optional forces the user to explicitly check if a value is present before 
 *   accessing it, thus minimizing the risk of null pointer dereference errors - it can also be 
 *   useful when designing APIs that can return optional values
 *
 * Some practical use cases of std::optional are
 *   1. Optional function arguments
 *   2. Configurations and settings 
 *   3. As the return value of an object that may fail 
 *
 * If an optional<T> contains a value, the value is guaranteed to be allocated as part of the
 *   optional object footprint - an optional object models an object, not a pointer
 *
 * A program is ill-formed if it instantiates an optional with a reference type, or with tag types
 */

#include <iostream>
#include <optional>

void printO(const std::optional<std::string>& o) {
    if (o.has_value()) {
        std::cout << "Value: " << o.value() << std::endl;
    } else {
        std::cout << "No Value" << std::endl;
    }
}

std::optional<std::string> create(bool b) {
    return b ? std::optional<std::string> {"str"} : std::nullopt;
}

int main() {
    std::optional<std::string> o;
    printO(o);

    o = "o1";
    printO(o);

    o = std::nullopt;
    printO(o);

    o = "o2";
    printO(o);

    o.reset();
    printO(o);

    printO(create(true));
    printO(create(false));
}
