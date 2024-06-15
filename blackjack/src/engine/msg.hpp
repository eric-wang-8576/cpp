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
    // Game state
    Hand const* dealerHand;
    std::vector<Hand*> const* playerHands;
    uint8_t playerIdx;

    // Display state
    std::string prevActionConfirmation;
    uint32_t stackSize;

    bool showBoard;

    bool prompt;
    std::string actionPrompt;

    bool gameOver;
    bool betInit;

    void print();
};
