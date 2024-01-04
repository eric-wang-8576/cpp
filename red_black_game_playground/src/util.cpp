#include "util.hpp"

namespace GameUtil {
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
        
    // Initialize a random number generator with the seed
    std::mt19937 generator(seed);

    int minBet = 1;
    int maxBet = 2;

    // Returns a random integer from 0 to max, non-inclusive
    int generateRandom(int max) {
        return generator() % max;
    }
}
