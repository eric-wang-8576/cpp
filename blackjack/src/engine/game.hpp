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
    uint32_t numShuffles;

    // Money State
    uint32_t buyIn;
    uint32_t stackSize;
    uint32_t tips;

    // Hand State
    Shoe shoe;
    Hand dealerHand;
    uint8_t numPlayerHands;
    Hand playerHands[256];
    // playerIdx refers to the current idx in playerHands the user needs to act on
    uint8_t playerIdx;

    // Bet State
    uint32_t prevBet;

    bool gameOver;

    // Copy Board Status
    void fillMsg(Msg& msg);

    // Handlers
    void handleAdd(uint32_t addValue, Msg& msg);
    void handleBet(uint32_t betAmt, Msg& msg);
    void handleHit(Msg& msg);
    void handleStand(Msg& msg);
    void handleDouble(Msg& msg);
    void handleSplit(Msg& msg);

    void concludeHand(Msg& msg);
    
    void resetBoard();

public:
    Game(int numDecks) : 
        activeBoard(false),
        numHands(0),
        buyIn(0), 
        stackSize(0), 
        playerIdx(0), 
        numPlayerHands(0), 
        shoe(numDecks),
        prevBet(100),
        tips(0),
        numShuffles(0),
        gameOver(false) {}

    // API for user
    // Filters user requests for correct usage corresponding to game state
    void processInput(std::string input, Msg& msg); 

};
