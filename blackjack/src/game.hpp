#pragma once

#include <regex>

#include "hand.hpp"
#include "shoe.hpp"
#include "msg.hpp"

enum GameState {
    INHAND,
    NOTINHAND,
};
    
class Game {
private:
    // Game State
    bool activeBoard;

    // Money State
    uint32_t buyIn;
    uint32_t stackSize;
    uint32_t tips;

    // Hand State
    uint32_t handNum;
    Hand dealerHand;
    std::vector<Hand> playerHands;

    // Bet State
    uint8_t prevBet;

    Shoe shoe;

public:
    Game() : 
        activeBoard(false),
        buyIn(500), 
        stackSize(500), 
        tips(0), 
        handNum(0),
        shoe(6),
        prevBet(100) {}

    Msg processInput(std::string input);
    
    void resetBoard();
};
