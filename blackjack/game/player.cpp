#include <iostream>

#include "game.hpp"

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

int main(int numArgs, char** argv) {
    if (numArgs != 2) {
        std::cout << "Please specify the number of decks" << std::endl;
        exit(1);
    }

    int numDecks = std::atoi(argv[1]);

    std::cout << c_yellow;
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
    std::cout << white;

    std::cout << "\nUse the following commands. To see them again, type \'help\'.\n\n"
              << commands << std::endl;
    
    std::cout << c_purple;
    std::cout << "\t♦♦♦  $0  ♦♦♦\n\n" << std::endl;

    Game game {numDecks};
    std::string userInput;
    Msg msg;
    while (true) {
        std::cout << c_cyan;
        std::cout << "ACTION ❯ ";

        std::getline(std::cin, userInput);
        if (userInput == "help") {
            std::cout << commands << std::endl;
        } else {
            game.processInput(std::move(userInput), msg);
            msg.print();
            if (msg.gameOver) {
                exit(0);
            }
        }
    }
}

