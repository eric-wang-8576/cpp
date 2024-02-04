#include "hand.hpp"

void Hand::addCard(Card card) {
    cards.push_back(card);
    updateValues();
}

void Hand::updateValues() {
    // set this->values
    values.clear();

    uint8_t numAces = 0;
    uint8_t total = 0;
    for (Card& card : cards) {
        total += card.getVal();
        if (card.getVal() == 11) {
            numAces++;
        }
    }

    for (uint8_t ace = 0; ace <= numAces; ace++) {
        uint8_t currVal = total - ace * 10;
        if (currVal <= 21) {
            values.push_back(currVal);
        }
    }

    std::reverse(values.begin(), values.end());
    
    // set this->busted
    busted = values.size() == 0 ? true : false;

    // set this->isBlackJack
    if ( (cards.size() == 2) &&
         ( (cards[0].getVal() == 11 && cards[1].getVal() == 10) || 
            (cards[0].getVal() == 10 && cards[1].getVal() == 11) )
       ) 
    {
        isBlackJack = true;
    } else {
        isBlackJack = false;
    }
}

std::string Hand::getString() {
    std::string str = "";

    // 12 magic number is maximum # possible without busting
    bool totalObscured = false;
    for (int i = 0; i < 12; ++i) {
        if (cards[i].isObscured()) {
            totalObscured = true;
        }
        if (i < cards.size()) {
            str += cards[i].getString();
        } else {
            str += "  ";
        }
        str += " ";
    }

    str += "(";
    if (busted) {
        str += "BUST";

    } else if (totalObscured) {
        str += "?";

    } else if (isBlackJack) {
        str += "BLACKJACK";

    } else {
        uint8_t numValues = values.size();
        for (int idx = 0; idx < numValues; ++idx) {
            str += std::to_string(values[idx]);
            if (idx != numValues - 1) {
                str += "/";
            }
        }
    }
    str += ")";

    if (betAmt != 0) {
        str += "          $" + std::to_string(betAmt);
    }

    return str;
}
