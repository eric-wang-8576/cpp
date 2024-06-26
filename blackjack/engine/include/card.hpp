#pragma once
#include <cassert>
#include <string>

// cardID ranges from 1 to 13, with 1 representing ace
// cardID of 0 is the null card
class Card {
    uint8_t cardID;
    
public:
    Card(uint8_t v) : cardID(v) {}

    // Swap function
    friend void swap(Card& a, Card& b) noexcept;

    // Returns the card as a string
    std::string getString();

    // Returns 11 for ace and 10 for paint
    uint8_t getVal();
    uint8_t getID();
};
