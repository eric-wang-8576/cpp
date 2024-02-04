#include "game.hpp"

std::regex addPattern("a\\s\\$\\d+");

Msg Game::processInput(std::string input) {
    Msg msg;

    if (std::regex_match(input, addPattern)) {
        
        uint32_t addValue = std::stoi(input.substr(3));
        buyIn += addValue;
        stackSize += addValue;

        msg.prevActionConfirmation = "You have added $" + 
                                     std::to_string(addValue) + 
                                     " to your stack. Your new stack size is $" +
                                     std::to_string(stackSize) + 
                                     ".";

    } else {
        msg.prevActionConfirmation = "Invalid Action -> Please Try Again";
        // Add Detailed Response Here 
    }

    return msg;
}
