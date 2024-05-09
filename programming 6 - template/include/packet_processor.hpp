#pragma once
#include "fingerprint.hpp"
#include "packet_generator.hpp"
#include "interval_tracker.hpp"
#include <set>
#include <mutex>

#define NUMLOCKS 32

class PacketProcessor {
    int numBins;
    int maxVal;
    std::vector<int> histogram;
    FingerPrint f;

    std::map<int, bool> PNG;
    std::map<int, IntervalTracker> R;
    int numLocks;
    std::vector<std::mutex> PNGlocks;
    std::vector<std::mutex> Rlocks;

public:
    PacketProcessor(int numBinsP, int numAddressesLog) 
        : numBins(numBinsP), maxVal(1 << numAddressesLog), numLocks(NUMLOCKS),
          PNGlocks(NUMLOCKS), Rlocks(NUMLOCKS)
    {
        histogram.resize(numBins, 0);
    }
        
    void processPacket(Packet& packet);
    void dumpHistogram();
};



