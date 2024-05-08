#include <fingerprint.hpp>

long FingerPrint::getFingerprint(long iterations, long startSeed) {
    long seed = startSeed;
    for (long i = 0; i < iterations; i++) {
        seed = (seed * a + c) & m;
    }
    return (seed >> 12) & 0xFFFFL;
}

