#include <iostream>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"


std::string commands = std::string{} + "\n" + 
    "- \'e\': exits the game\n" +
    "- \'b\': places bet equal to the previous one (default $100)\n" +
    "- \'h\': hit\n" +
    "- \'s\': stand\n" + 
    "- \'d\': double down\n" + 
    "- \'p\': split\n" +
    "- \'a $XXX\': adds on $XXX to your stack\n" +
    "- \'b $XXX\': places a bet of $XXX\n" +
    "- \'t $XXX\': tips the dealer $XXX\n\n";

int main() {

    std::cout << std::endl;
    for (int i = 0; i < 35; ++i) {
        std::cout << "~";
    }
    std::cout << std::endl;
    std::cout << "Welcome to Command Line Blackjack!" << std::endl;
    for (int i = 0; i < 35; ++i) {
        std::cout << "~";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "\nUse the following commands. To see them again, type \'help\'.\n\n"
              << commands << std::endl;
    
    std::cout << "\t♦♦♦  $0  ♦♦♦\n\n" << std::endl;

    Game game;
    std::string userInput;
    while (true) {
        std::cout << "ACTION ❯ ";

        std::getline(std::cin, userInput);
        if (userInput == "help") {
            std::cout << commands << std::endl;
        } else {
            Msg msg = game.processInput(std::move(userInput));
            msg.print();
            if (msg.gameOver) {
                exit(0);
            }
        }
    }
}

