#pragma once
#include <cassert>
#include <string>

// val ranges from 1 to 13, with 1 representing ace
class Card {
    uint8_t val;
    
public:
    Card(uint8_t v) : val(v) {
        assert(1 <= v && v <= 13);
    }
    std::string getString();
    // returns 11 for ace and 10 for paint
    uint8_t getVal();
};
