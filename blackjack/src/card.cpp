#include "card.hpp"

std::string Card::getString() {
    if (obscured) {
        return "??";
    }
    switch (val) {
        case 1:
            return "A ";
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
            return std::to_string(val) + " ";
        case 10:
            return "10";
        case 11:
            return "J ";
        case 12:
            return "Q ";
        case 13:
            return "K ";
        default:
            return "";
    }
};

// Returns 11 for Ace
uint8_t Card::getVal() {
    if (val == 1) {
        return 11;
    } else if (val < 10) {
        return val;
    } else {
        return 10;
    }
};

void Card::setObscured(bool b) {
    obscured = b;
}

bool Card::isObscured() {
    return obscured;
}
