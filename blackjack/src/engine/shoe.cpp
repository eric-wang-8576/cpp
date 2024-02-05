#include "shoe.hpp"

Card Shoe::draw() {
    if (cards.size() < 16) {
        shuffle();
    }

    // Potentially std::move() this, but the int is copied anyways 
    Card ret = cards.back();
    cards.pop_back();

    return ret;
}

void Shoe::shuffle() {
    cards.clear();

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
