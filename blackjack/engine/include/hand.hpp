#pragma once

#include "card.hpp"
#include "util.hpp"
#include <iostream>

class Hand {
    // Only 26 cards are possible in any given hand
    int numCards;
    Card* cards[26];

    // Only two values are possible at any time
    // If there are two values, the greater value will
    // be stored first
    int numValues;
    uint8_t values[2];

    // Hand states
    bool obscured;
    bool isOriginal; // original two cards dealt by dealer 

    // Amount of money the user bet on this hand
    uint32_t betAmt;

    void updateValues();

public:

    Hand() : numCards(0),
             numValues(0),
             obscured(false), 
             isOriginal(false),
             betAmt(0) {}

    // Modify hand cards
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
    uint32_t getBetAmt();
    uint8_t getFirstCardValue();
    bool isObscured();
    uint8_t getCardID(uint8_t idx);

    // Compute if the dealer should hit this hand
    bool shouldDealerHit();

    // Reset the dealer's hand
    void reset();

    void setBetAmt(uint32_t amt);
    void setObscured(bool obs);
    void setOriginal(bool orig);

    std::string getString();
};


    
    
