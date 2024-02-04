#pragma once
#include <cassert>
#include <string>

// val ranges from 1 to 13, with 1 representing ace
class Card {
    uint8_t val;
    bool obscured;
    
public:
    Card(uint8_t v) : val(v), obscured(false) {
        assert(1 <= v && v <= 13);
    }
    std::string getString();
    // returns 11 for ace and 10 for paint
    uint8_t getVal();

    void setObscured(bool b);
    bool isObscured();
};
