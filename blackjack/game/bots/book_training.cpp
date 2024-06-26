#include <iostream>

#include "game.hpp"
#include "strategy.hpp"

void printBoldError(const std::string& message) {
    std::string bold = "\033[1m";  // ANSI code for bold
    std::string red = "\033[31m";  // ANSI code for red
    std::string reset = "\033[0m"; // ANSI code to reset formatting

    std::string top_bottom_border = "###############################";
    std::string side_border = "##";

    // Print top border
    std::cout << std::endl;
    std::cout << red << bold << top_bottom_border << reset << std::endl;

    // Print message with side borders
    std::cout << red << bold << side_border << " " << message << " " << side_border << reset << std::endl;

    // Print bottom border
    std::cout << red << bold << top_bottom_border << reset << std::endl;
    std::cout << std::endl;
}

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

    // TODO: Fix this code
    int numDecks = (int) (*argv[1] - '0');

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
            if ((userInput == "h" || userInput == "d" || userInput == "p" || userInput == "s") && userInput != Strategy::generateAction(msg))
            {
                if (msg.actionPrompt != "Option: bet") 
                {

                    printBoldError("WROOOOOOOOOOOOOOOOONG !!!");
                    continue;
                }
            }

            game.processInput(std::move(userInput), msg);
            msg.print();
            if (msg.gameOver) {
                exit(0);
            }
        }
    }
}

