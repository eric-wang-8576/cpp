#pragma once

#include "card.hpp"

class Game {
private:
    // Money State
    uint32_t buyIn;
    uint32_t stackSize;
    uint32_t PNL;
    uint32_t tips;

    // Hand State
    uint32_t handNum;
};
