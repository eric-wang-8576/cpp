#pragma once

#include <random>
#include <iostream>
#include <ctime>

enum COLOR {
    RED,
    BLACK,
};

namespace GameUtil {
    extern int minBet;
    extern int maxBet;

    extern unsigned int seed;

    extern std::mt19937 generator;

    int generateRandom(int);
}
