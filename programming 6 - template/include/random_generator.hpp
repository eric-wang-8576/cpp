#pragma once
#include <random>
#include <cmath>

class RandomGenerator;
class UniformGenerator;
class ExponentialGenerator;

class RandomGenerator {
    int seed;
    
public:
    RandomGenerator(int seed) {
        this->seed = seed;
    }

    RandomGenerator() {
        this->seed = 59009;
    }

    int getRand();
    void setSeed(int seed);
    int mangle(int seed);
};

class UniformGenerator {
    RandomGenerator randGen;

public:
    UniformGenerator(int seed) : randGen(seed) {}

    UniformGenerator() {}

    int getRand(int minValue, int maxValue);
    int getRand(int maxValue);
    int getRand();
    double getUnitRand();
    void setSeed(int startSeed);
    int mangle(int seed);
};

class ExponentialGenerator {
    double mean;
    RandomGenerator randGen;
    const double logBase = 20.795; // ln(2^30 - 1)
    const int base = 1073741824; // 2^30
                                 //
public:
    ExponentialGenerator(double mean) {
        this->mean = mean;
    }

    int getRand();
    int getRand(double meanTmp);
    int mangle(int seed);
};

