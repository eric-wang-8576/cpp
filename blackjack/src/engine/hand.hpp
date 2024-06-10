#pragma once

#include "card.hpp"

class Hand {
    // contains the possible <= 21 values of the hand
    
    // player controlled
    bool busted;

    void updateValues();

public:
    bool obscured;
    bool isBlackJack;
    std::vector<uint8_t> values;
    std::vector<Card> cards;
    uint32_t betAmt;// TODO: Make this private

    Hand() : obscured(false), busted(false), isBlackJack(false), betAmt(0) {}

    // adds the card and updates internal values
    void addCard(Card card);
    void popCard();

    void setFinished();
    bool isFinished();

    bool isBusted();

    bool isSoft();
    
    bool isPair();

    bool shouldDraw();

    int getLastVal();

    std::string getString();
};


    
    
