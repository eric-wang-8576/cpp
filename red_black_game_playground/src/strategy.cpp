#include "game.hpp"
#include "strategy.hpp"

void Strategy::play() {
    while (!game->isFinished()) {
        COLOR color;
        int betQty;

        if (numRed == numBlack) {
            color = COLOR::RED;
            betQty = GameUtil::minBet;
        } else if (numRed < numBlack) {
            color = COLOR::BLACK;
            betQty = GameUtil::maxBet;
        } else {
            color = COLOR::RED;
            betQty = GameUtil::maxBet;
        }

        COLOR res = game->draw(color, betQty);
        if (res == COLOR::RED) {
            numRed--;
        } else {
            numBlack--;
        }
    }
}

void Strategy::random() {
    while (!game->isFinished()) {
        COLOR color = GameUtil::generateRandom(2) == 1 ? COLOR::RED : COLOR::BLACK;
        int amount = 1 + GameUtil::generateRandom(GameUtil::maxBet);
        game->draw(color, amount);
    }
}
