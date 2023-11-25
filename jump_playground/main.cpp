#include <iostream>
#include <fstream>
#include <thread>
#include <queue>
#include <condition_variable>

template<typename ... Args>
void printString(const char* str, Args ... args) {
    auto size = std::snprintf(nullptr, 0, str, args...) + 1;
    char newStr[size];
    std::snprintf(newStr, size, str, args...);
    printf("%s", newStr);
}

int main() {
    printString("Hi, %s", "Eric");
    printString("We know that %d + %d = %d", 4, 5, 9);
    return 0;
}
