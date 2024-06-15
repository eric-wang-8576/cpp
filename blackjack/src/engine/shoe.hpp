#pragma once

#include <random>
#include <algorithm>
#include <iostream>

#include "card.hpp"

/*
 * this->cards contains a vector of the current cards in the shoe
 * Cards are removed when they are drawn
 * On a shuffle, the current cards are discarded, and this->cards
 * is replaced with a newly shuffled set of cards
 */

class Shoe {
    uint8_t numDecks;
    // Refers to the next card to draw
    uint8_t cardIdx;
    uint8_t numCards;
    std::vector<Card> cards;

    std::random_device rd_;
    std::mt19937 g_;

public:
    Shoe(uint8_t n) : numDecks(n), 
                      numCards(n * 52),
                      rd_(), 
                      g_(rd_())
    {
        cards.reserve(numCards);
        for (uint8_t deck = 0; deck < numDecks; deck++) {
            for (uint8_t suit = 0; suit < 4; ++suit) {
                for (uint8_t cardID = 1; cardID <= 13; ++cardID) {
                    cards.emplace_back(cardID);
                }
            }
        }

        shuffle();
    }

    // Draws a card and returns a reference to it
    Card* draw();

    // Shuffles the deck if we have used up 2/3 of the cards
    // and returns if the deck was shuffled
    bool triggerShuffle();
    void shuffle();
};
