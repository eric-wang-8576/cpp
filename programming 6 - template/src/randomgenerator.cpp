#include "randomgenerator.hpp"

int RandomGenerator::getRand() {
    seed = mangle(seed) + 1;
    return seed;
}

void RandomGenerator::setSeed(int seed) {
    this->seed = seed;
}

int RandomGenerator::mangle(int seed) { 
    int CRC_POLY = 954680065; // 0x38E74301 - standard CRC30 from CDMA
    int iterations = 31;
    int crc = seed;
    for( int i = 0; i < iterations; i++ ) {
      if( ( crc & 1 ) > 0 )
        crc = (crc >> 1) ^ CRC_POLY;
      else
        crc = crc >> 1;
    }
    return crc;
}

int UniformGenerator::getRand(int minValue, int maxValue) { // [minValue,maxValue) like indexing
    return ( randGen.getRand()  % (maxValue-minValue) ) + minValue;
}

int UniformGenerator::getRand(int maxValue) { // [0,maxValue)
    return randGen.getRand() % maxValue;
}

int UniformGenerator::getRand() {
    return randGen.getRand();
}

double UniformGenerator::getUnitRand() {
    double base_d = 1073741824.0; // 2^30
    int base_i = 1073741824; // 2^30
    return ((double) getRand(base_i))/base_d;
}

void UniformGenerator::setSeed(int startSeed) {
    randGen.setSeed(startSeed);
}

int UniformGenerator::mangle(int seed) { return randGen.mangle(seed); }

int ExponentialGenerator::getRand() {
    return (int) std::ceil(mean*(logBase-std::log(base-randGen.getRand())));
}

int ExponentialGenerator::getRand(double meanTmp) {
    return (int) std::ceil(meanTmp*(logBase-std::log(base-randGen.getRand())));
}

int ExponentialGenerator::mangle(int seed) { return randGen.mangle(seed); }

