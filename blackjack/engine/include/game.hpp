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
    std::smatch matches;

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
    uint32_t numBets;
    uint32_t totalBet;
    uint32_t bets[7];

    bool gameOver;

    // Copy Board Status
    void fillMsg(Msg& msg);

    // Parsers
    bool parseAdd(const std::string& input);
    bool parseBet(const std::string& input);
    bool parseTip(const std::string& input);

    // Handlers
    void handleAdd(uint32_t addValue, Msg& msg);
    void handleBet(Msg& msg);
    void handleHit(Msg& msg);
    void handleStand(Msg& msg);
    void handleDouble(Msg& msg);
    void handleSplit(Msg& msg);
    void handleTip(uint32_t tipValue, Msg& msg);
    void handleExit(Msg& msg);


    void advancePlayerIdx();

    void concludeHand(Msg& msg, bool justShuffled = false);
    
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
        numBets(1),
        tips(0),
        numShuffles(0),
        gameOver(false),
        totalBet(100) {
        bets[0] = 100;

    }

    // API for user
    // Filters user requests for correct usage corresponding to game state
    void processInput(const std::string& input, Msg& msg); 

};
