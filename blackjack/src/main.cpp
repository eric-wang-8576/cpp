#include <iostream>

#include "game.hpp"
#include "shoe.hpp"

int main() {
    Shoe shoe {6};
    for (int i = 0; i < 10000; ++i) {


        std::cout << shoe.draw().getString() << " ";
    }
}

