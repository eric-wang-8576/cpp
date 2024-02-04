#include <iostream>

#include "game.hpp"
#include "shoe.hpp"
#include "hand.hpp"

int main() {

//    Shoe shoe {6};
//    
//    for (int i = 0; i < 100; ++i) {
//        Hand hand;
//        for (int i = 0; i < 2; ++i) {
//            hand.addCard(shoe.draw());
//        }
//        std::cout << hand.getString() << std::endl;
//        std::cout << std::endl;
//    }

    Game game;
    std::string userInput;
    while (true) {
        std::cout << "ACTION ❯ ";

        std::getline(std::cin, userInput);
        Msg msg = game.processInput(std::move(userInput));
        msg.print();
    }
}

