#pragma once

#include <random>
#include <algorithm>

#include "card.hpp"

/*
 * this->cards contains a vector of the current cards in the shoe
 * Cards are removed when they are drawn
 * On a shuffle, the current cards are discarded, and this->cards
 * is replaced with a newly shuffled set of cards
 */

class Shoe {
    uint8_t numDecks;
    std::vector<Card> cards;

public:
    int count;

    Shoe(uint8_t n) : numDecks(n) {}
    Card draw();
    void shuffle();
};
