#include "msg.hpp"

std::string actionBorder {"--------------"};
std::string boardBorder {"~~~~~~~~~~~~~~~~~~~~~~~"};

std::string bold = "\033[1m";  
std::string red = "\033[31m"; 
std::string green = "\033[32m";  
std::string yellow = "\033[33m";
std::string blue = "\033[34m";
std::string magenta = "\033[35m";
std::string cyan = "\033[36m";
std::string white = "\033[37m";
std::string custom = "\033[38;5;m";

std::string bold_on = "\033[1m";
std::string bold_off = "\033[0m";

std::string c_purple = "\033[38;5;129m";
std::string c_cyan = "\033[38;5;51m";
std::string c_yellow = "\033[38;5;11m";
std::string c_orange = "\033[38;5;214m";
std::string c_red = "\033[38;5;204m";

std::string reset = "\033[0m"; // ANSI code to reset formatting

void Msg::print() {

    if (betInit) {
        std::cout << white;
        std::cout << std::endl;
        for(int i = 0; i < 80; ++i) {
            std::cout << "*";
        }
        std::cout << reset << std::endl;
    }

    std::cout << std::endl;

    std::cout << c_yellow;

    std::cout << actionBorder << std::endl;
    std::cout << prevActionConfirmation << std::endl;
    std::cout << actionBorder << std::endl;

    std::cout << std::endl;

    if (showBoard) {
        std::cout << green;
        std::cout << boardBorder << std::endl;
        std::cout << "|| BOARD ||" << std::endl;
        std::cout << "Dealer's Hand:        " << dealerHandP->getString() << std::endl;
        for (int hand = 0; hand < numPlayerHands; ++hand) {
            if (hand == playerIdx) {
                std::cout << bold_on;
            }

            std::cout << "Player's Hand #" << hand + 1 << ":    ";
            if (hand < 9) {
                std::cout << " ";
            }
            std::cout << playerHandsP[hand].getString() << std::endl;

            if (hand == playerIdx) {
                std::cout << bold_off << green;
            }
        }
        std::cout << boardBorder << std::endl;
    }

    std::cout << c_purple;

    std::cout << "\n\t♦♦♦  " << priceToString(stackSize) << "  ♦♦♦\n" << std::endl;

    std::cout << c_cyan;

    if (prompt) {
        std::cout << std::endl;
        std::cout << "---> " << actionPrompt;
        std::cout << std::endl;
    }

    for (uint8_t i = 0; i < 5; ++i) {
        std::cout << std::endl;
    }
}
