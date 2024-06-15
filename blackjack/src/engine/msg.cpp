#include "msg.hpp"

std::string actionBorder {"--------------"};
std::string boardBorder {"~~~~~~~~~~~~~~~~~~~~~~~"};

void Msg::print() {

    if (betInit) {
        std::cout << std::endl;
        for(int i = 0; i < 80; ++i) {
            std::cout << "*";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    std::cout << actionBorder << std::endl;
    std::cout << prevActionConfirmation << std::endl;
    std::cout << actionBorder << std::endl;

    std::cout << std::endl;

    if (showBoard) {
        std::cout << boardBorder << std::endl;
        std::cout << "|| BOARD ||" << std::endl;
        std::cout << "Dealer's Hand:       " << dealerHand.getString() << std::endl;
        for (int hand = 0; hand < playerHands.size(); ++hand) {
            std::cout << "Player's Hand #" << hand + 1 << ":    " <<
                playerHands[hand].getString() << std::endl;
        }
        std::cout << boardBorder << std::endl;
    }

    std::cout << "\n\t♦♦♦  $" << stackSize << "  ♦♦♦\n" << std::endl;

    if (prompt) {
        std::cout << std::endl;
        std::cout << "---> " << actionPrompt;
        if (!dealerHand.obscured) {
            std::cout << "\t[ " << count << " ]";
        }
        std::cout << std::endl;
    }

    for (uint8_t i = 0; i < 5; ++i) {
        std::cout << std::endl;
    }
}
