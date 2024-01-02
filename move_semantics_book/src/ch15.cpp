/*
 * Moved-from standard strings are usually empty but that is not guaranteed
 *
 * Moved-from standard containers (except std::array<>) are usually empty if no special allocator is 
 *   used - for vectors, this is indirectly guaranteed - for othe other containers, it is indirectly
 *   guaranteed that all elements are moved away or destroyed and inserting new members would make no 
 *   sense
 *
 * Move assignments can change the capacity of strings and vectors
 *
 * To support decaying when passing values to universal/forwarding references (often necessary to 
 *   deduce the same type for string literals of different length), use the type trait std::decay()
 *
 * Generic wrapper types should use reference qualifiers to overload member functions that access the
 *   wrapped objects - this might even mean using overloads for const rvalue references 
 *
 * Avoid copying of shared pointers (by passing them by value)
 *
 * Use std::jthread (available since C++20) instead of std::thread
 */

/*
 * Some platforms may swap memory, while others may move memory instead when performing std::move()
 *   with dynamically allocated memory 
 *
 * All containers support move semantics when copying the containers, assigning the containers, or 
 *   inserting elements into the container 
 *
 * The C++ standard specifies constant complexity for a move constructor for a container - for vectors,
 *   providing a value in a moved-from object is even indirectly forbidden because the move constructor
 *   guarantees never to throw 
 *
 * The standard also specifies that a move operation either overwrites or destroys each element of the 
 *   destination object - for containers after using them as a source in a move assignment, we are 
 *   guaranteed that the container is empty if the memory is interchangeable, or valid but unspecified
 *   otherwise
 *
 * The function emplace_back() directly initializes new elements in the container, using perfect
 *   forwarding
 *
 * Moving an std::array is better than copying if we can move the elements
 *
 * The class std::optional<> is one of the rare places in C++ where const rvalue references are used - 
 *   this is because std::optional<> is a wrapper type that wants to ensure that the operations do the
 *   right thing even when const objects are marked with std::move() 
 *
 * A function binding a value parameter to an rvalue takes ownership of it, while a function binding
 *   an lvalue reference to an rvalue may or may not take ownership of it
 *
 * By default, the constructor of the thread class copies the arguments - therefore, we can utilize
 *   move semantics with the parameters
 */

// NOTE: std::pair, std::optional, and std::promise/std::future code are in separate playgrounds

//    template<typename... Args>
//    constexpr T& emplace_back(Args&&... args) {
//        place_element_in_memory(T(std::forward<Args>(args)...));
//    }

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

std::ofstream openToWrite(const std::string& name) {
    std::ofstream file(name);
    if (!file) {
        std::cerr << "Can't Open File " << name << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return file;
}

// If we use std::ofstream&& instead, it creates an rvalue reference and does not take ownership
void storeData(std::ofstream fstrm) {
    fstrm << 42 << std::endl;
}

std::unique_ptr<std::string> create() {
    static long id = 0;
    auto ptr = std::make_unique<std::string>("obj #" + std::to_string(++id));
    return ptr;
}

void doThis(const std::string& arg) {
    std::cout << "doThis(): " << arg << std::endl;
}

void doThat(const std::string& arg) {
    std::cout << "doThat(): " << arg << std::endl;
}

int main() {
    std::unique_ptr<std::string> p;
    for (int i = 0; i < 10; ++i) {
        p = create();
        std::cout << *p << std::endl;
    }

    auto outFile {openToWrite("iostream.tmp")};
    storeData(std::move(outFile));
    if (outFile.is_open()) {
        std::cout << "Closing file" << std::endl;
        outFile.close();
    }

    std::vector<std::thread> threads;

    std::string arg1 {"arg1"};
    threads.push_back(std::thread{doThis, arg1});
    threads.push_back(std::thread{doThat, std::move(arg1)});
    
    for (auto& t : threads) {
        t.join();
    }

    std::cout << arg1 << std::endl;
}
