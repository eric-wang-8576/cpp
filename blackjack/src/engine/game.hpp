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
    uint32_t numHands;

    // Money State
    uint32_t buyIn;
    uint32_t stackSize;
    uint32_t tips;

    // Hand State
    Hand dealerHand;
    std::vector<Hand> playerHands;
    uint8_t playerIdx;
    

    // Bet State
    uint32_t prevBet;

    bool gameOver;

    Shoe shoe;

public:
    Game(int numDecks) : 
        activeBoard(false),
        numHands(0),
        buyIn(0), 
        stackSize(0), 
        shoe(numDecks),
        prevBet(100),
        tips(0),
        gameOver(false) {}

    // Copy Board Status
    void fillMsg(Msg& msg);

    // Handlers
    Msg handleAdd(uint32_t addValue);
    Msg handleBet(uint32_t betAmt);
    Msg handleHit();
    Msg handleStand();
    Msg handleDouble();
    Msg handleSplit();

    Msg processInput(std::string input); 
    Msg concludeHand();
    
    void resetBoard();
};
