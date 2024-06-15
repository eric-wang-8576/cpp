#include "hand.hpp"

void Hand::updateValues() {
    uint8_t numAces = 0;
    uint8_t total = 0;
    for (int i = 0; i < numCards; ++i) {
        total += cards[i]->getVal();
        if (cards[i]->getVal() == 11) {
            numAces++;
        }
    }

    // Compute potential values
    numValues = 0;
    for (uint8_t ace = 0; ace <= numAces; ace++) {
        uint8_t currVal = total - ace * 10;
        if (currVal <= 21) {
            values[numValues] = currVal;
            numValues++;
        }
    }

    // Set busted
    busted = numValues == 0 ? true : false;

    // set this->isBlackJack
    if ( (numCards == 2) &&
         ( (cards[0]->getVal() == 11 && cards[1]->getVal() == 10) || 
            (cards[0]->getVal() == 10 && cards[1]->getVal() == 11) )
       ) 
    {
        isBlackJack = true;
    } else {
        isBlackJack = false;
    }
}

void Hand::addCard(Card* cardP) {
    cards[numCards] = cardP;
    numCards++;
    updateValues();
}

Card* Hand::popCard() {
    Card* cardP = cards[numCards - 1];
    numCards--;
    updateValues();
    return cardP;
}

std::string Hand::getString() {
    std::string str = "";

    // 12 magic number is maximum # possible without busting
    for (int i = 0; i < 12; ++i) {
        if (i == 1 && obscured) { // obscure dealer's second card
            str += "??";
        } else if (i < numCards) {
            str += cards[i]->getString();
        } else {
            str += "  ";
        }
        str += " ";
    }

    str += "(";
    if (busted) {
        str += "BUST";

    } else if (obscured) {
        str += "?";

    } else if (isBlackJack) {
        str += "BLACKJACK";

    } else {
        for (int idx = numValues - 1; idx >= 0; --idx) {
            str += std::to_string(values[idx]);
            if (idx != 0) {
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

bool Hand::isBusted() {
    return busted;
}

bool Hand::isSoft() {
    return numValues == 2;
}

bool Hand::isPair() {
    return (numCards == 2 && (cards[0]->getVal() == cards[1]->getVal()));
}

bool Hand::isBlackjack() {
    return isBlackJack;
}

bool Hand::areAces() {
    return (numCards == 2 && cards[0]->getVal() == 11 && cards[1]->getVal() == 11);
}

uint8_t Hand::getPrimaryVal() {
    return values[0];
}

uint8_t Hand::getNumCards() {
    return numCards;
}

uint32_t Hand::getBetAmt() {
    return betAmt;
}

uint8_t Hand::getFirstCardValue() {
    return cards[0]->getVal();
}

bool Hand::isObscured() {
    return obscured;
}

bool Hand::shouldDealerHit() {
    // Soft 17
    if (numValues == 2 && (values[0] == 17) && (values[1] == 7)) {
        return true;
    }
    return (!busted) && (values[0] < 17);
}

void Hand::reset() {
    numCards = 0;
    numValues = 0;
}

void Hand::setBetAmt(uint32_t amt) {
    betAmt = amt;
}

void Hand::setObscured(bool obs) {
    obscured = obs;
}

