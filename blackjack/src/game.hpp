#pragma once

#include <regex>

#include "hand.hpp"
#include "shoe.hpp"
#include "msg.hpp"

class Game {
private:
    // Game State
    

    // Money State
    uint32_t buyIn;
    uint32_t stackSize;
    uint32_t tips;

    // Hand State
    uint32_t handNum;
    Hand dealerHand;
    std::vector<Hand> playerHands;

    Shoe shoe;

public:
    Game() : 
        buyIn(0), 
        stackSize(0), 
        tips(0), 
        handNum(0),
        shoe(6) {}

    Msg processInput(std::string input);
};
