#pragma once

#include "card.hpp"

class Hand {
    // contains the possible <= 21 values of the hand
    
    // player controlled
    bool finished;
    bool isBlackJack;
    
    void updateValues();

public:
    std::vector<uint8_t> values;
    bool busted;
    std::vector<Card> cards;

    uint32_t betAmt;// TODO: Make this private

    Hand() : finished(false), busted(false), isBlackJack(false), betAmt(0) {}

    // adds the card and updates internal values
    void addCard(Card card);

    void setFinished();
    bool isFinished();

    bool isBusted();

    bool isSoft();

    std::string getString();
};


    
    
