#include <iostream>

// A function parameter pack is a function parameter that accepts zero or more function arguments
// A template with at least one parameter pack is called a variadic template

template<typename... Args>
void printThreeTimes(const char* str, Args&&... args) {
    for (int i = 0; i < 3; ++i) {
        printf(str, std::forward<Args>(args)...);
    }
}

void printAll() {
    std::cout << std::endl;
}

template<typename T, typename... A> 
void printAll(T t, A... a) {
    std::cout << t << " ";
    printAll(a...);
}

template<typename... Types>
class TuplePrinter {
public:
    TuplePrinter(const std::tuple<Types...>& t) : tuple(t) {}

    void print() {
        printTuple(std::make_index_sequence<sizeof...(Types)>{});
    }

private:
    std::tuple<Types...> tuple;

    template<std::size_t... Is>
    void printTuple(std::index_sequence<Is...>) {
        ((std::cout << std::get<Is>(tuple) << " "), ...);
        std::cout << std::endl;
    }
};

int main() {
    const char v[] = "Printing numbers %d and %d.\n";
    printThreeTimes(v, 5, 10);
    printAll(5, 10, 3.08, "hi!");

    auto t = std::make_tuple(1, 2.5, "Hi", 'a');
    TuplePrinter<int, double, const char*, char> printer(t);
    printer.print();

}
