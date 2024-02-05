#pragma once

#include <iostream>
#include "hand.hpp"

/*
 * All messages should have a prevActionConfirmation
 *
 * If prompt is true, then an actionPrompt will be conveyed
 *
 * playerIdx contains the index of the hand that the player should act on
 */
struct Msg {
    std::string prevActionConfirmation;
    uint32_t stackSize;

    bool showBoard;
    Hand dealerHand;
    std::vector<Hand> playerHands;
    uint8_t playerIdx;

    bool prompt;
    std::string actionPrompt;

    bool gameOver;

    void print();
};
