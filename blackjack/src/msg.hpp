#pragma once

#include <iostream>
#include "hand.hpp"

struct Msg {
    std::string prevActionConfirmation;
    uint32_t stackSize;

    bool showBoard;
    Hand dealerHand;
    std::vector<Hand> playerHands;

    bool prompt;
    std::string actionPrompt;

    bool gameOver;

    void print();
};
