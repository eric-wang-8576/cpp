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

    // Hand State
    Hand dealerHand;
    std::vector<Hand> playerHands;
    uint8_t playerIdx;
    

    // Bet State
    uint8_t prevBet;

    bool gameOver;

    Shoe shoe;

public:
    Game() : 
        activeBoard(false),
        buyIn(500), 
        stackSize(500), 
        shoe(6),
        prevBet(100),
        gameOver(false) {}

    // Copy Board Status
    void fillMsg(Msg& msg);

    // Handlers
    Msg handleAdd(uint32_t addValue);
    Msg handleBet(uint32_t betAmt);
    Msg handleHit();
    Msg handleStand();

    Msg processInput(std::string input); 
    Msg concludeHand();
    
    void resetBoard();
};
