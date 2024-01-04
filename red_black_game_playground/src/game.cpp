#include "game.hpp"

COLOR Game::draw(COLOR betColor, int betQty) {
    if (finished) {
        std::cout << "ERROR: Drawing on a finished game";
        exit(1);
    }

    if (betQty < minBet || betQty > maxBet) {
        std::cout << "ERROR: Incorrect bet size";
        exit(1);
    }

    int cardNum = GameUtil::generateRandom(numRed + numBlack);
    COLOR cardColor = cardNum < numRed ? COLOR::RED : COLOR::BLACK;

    if (cardColor == RED) {
        numRed--;
    } else {
        numBlack--;
    }
    if (numRed == 0 && numBlack == 0) {
        finished = true;
    }

    pnl += betColor == cardColor ? betQty : -betQty;
    return cardColor;
}

bool Game::isFinished() {
    return finished;
}

int Game::getPnl() {
    return pnl;
}
