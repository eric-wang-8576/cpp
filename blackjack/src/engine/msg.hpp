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
    Hand* dealerHandP;
    Hand* playerHandsP; // pointer to the start of an array
    uint8_t playerIdx;
    uint8_t numPlayerHands;

    // Display state
    std::string prevActionConfirmation;
    uint32_t stackSize;

    bool showBoard;

    bool prompt;
    std::string actionPrompt;

    bool gameOver;
    bool betInit;
    bool shuffled;

    void print();
};
