#include "shoe.hpp"

Card* Shoe::draw() {
    if (cardIdx == numCards) {
        std::cout << "Drawing on a finished shoe with cardIdx = " << cardIdx << std::endl;
        exit(1);
        return &cards[d_(g_)];
    }

    return &cards[cardIdx++];
}

bool Shoe::triggerShuffle() {
    // Shuffle the deck if we have used up 2/3 of the cards
    if (cardIdx > numCards * 2 / 3) {
        shuffle();
        return true;
    }
    
    return false;
}

void Shoe::shuffle() {
    std::shuffle(std::begin(cards), std::end(cards), g_);

    cardIdx = 0;
}
