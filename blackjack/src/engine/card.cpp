#include "card.hpp"

void swap(Card& a, Card& b) noexcept {
    std::swap(a.cardID, b.cardID);
}

std::string Card::getString() {
    switch (cardID) {
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
            return std::to_string(cardID) + " ";
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

uint8_t Card::getVal() {
    if (cardID == 1) {
        return 11;
    } else if (cardID < 10) {
        return cardID;
    } else {
        return 10;
    }
};
