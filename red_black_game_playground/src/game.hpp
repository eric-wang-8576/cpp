#pragma once

#include "util.hpp"

class Game {
private:
    int minBet;
    int maxBet;

    int pnl;
    int numRed;
    int numBlack;
    bool finished;

public:
    Game(int min, int max) :
        minBet(min), 
        maxBet(max), 
        pnl(0),
        numRed(26),
        numBlack(26),
        finished(false) {}

    COLOR draw(COLOR betColor, int betQty);
    bool isFinished();
    int getPnl();
};
