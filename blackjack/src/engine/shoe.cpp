#include "shoe.hpp"

Card Shoe::draw() {
    if (cards.size() < numDecks * 13) {
        shuffle();
    }

    // Potentially std::move() this, but the int is copied anyways 
    Card ret = cards.back();
    cards.pop_back();

    uint8_t val = ret.getVal();
    if (2 <= val && val <= 6) {
        count++;
    } else if (9 <= val && val <= 11) {
        count--;
    }

    return ret;
}

void Shoe::shuffle() {
    cards.clear();
    count = 0;

    for (uint8_t deck = 0; deck < numDecks; deck++) {
        // Add a deck to the shoe
        for (uint8_t suit = 0; suit < 4; ++suit) {
            for (uint8_t val = 1; val <= 13; ++val) {
                Card card {val};
                cards.push_back(card);
            }
        }
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);
    std::shuffle(std::begin(cards), std::end(cards), engine);
}
