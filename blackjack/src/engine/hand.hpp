#pragma once

#include "card.hpp"

class Hand {
    // Only 26 cards are possible in any given hand
    int numCards;
    std::array<Card*, 26> cards;

    // Only two values are possible at any time
    // If there are two values, the greater value will
    // be stored first
    int numValues;
    std::array<uint8_t, 2> values;

    // Hand states
    bool busted;
    bool obscured;
    bool isBlackJack;

    // Amount of money the user bet on this hand
    uint32_t betAmt;

    void updateValues();

public:

    Hand() : numCards(0),
             numValues(0),
             busted(false), 
             obscured(false), 
             isBlackJack(false), 
             betAmt(0) {}

    void addCard(Card* cardP);
    Card* popCard();

    // Check the state of the hand
    bool isBusted();
    bool isSoft();
    bool isPair();
    bool isBlackjack();
    bool areAces();
    uint8_t getPrimaryVal();
    uint8_t getNumCards();
    uint8_t getBetAmt();

    // Compute if the dealer should hit this hand
    bool shouldDealerHit();

    // Reset the dealer's hand
    void reset();

    void setBetAmt();



    std::string getString();
};


    
    
