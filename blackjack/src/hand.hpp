#pragma once

#include "card.hpp"

class Hand {
    std::vector<Card> cards;
    // contains the possible values of the hand, if there is no value
    // then the hand is busted
    std::vector<uint8_t> values;
    
    // player controlled
    bool finished;
    bool busted;
    bool isBlackJack;

    void updateValues();

public:
    Hand() : finished(false), busted(false), isBlackJack(false) {}

    // adds the card and updates internal values
    void addCard(Card card);

    void setFinished();
    bool isFinished();

    bool isBusted();

    bool isSoft();

    std::string getString();
};


    
    
