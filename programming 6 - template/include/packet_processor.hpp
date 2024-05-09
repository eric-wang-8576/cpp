#pragma once
#include "fingerprint.hpp"
#include "packet_generator.hpp"
#include "interval_tracker.hpp"
#include <set>

class PacketProcessor {
    int numBins;
    int maxVal;
    std::vector<int> histogram;
    FingerPrint f;

    std::map<int, bool> PNG;
    std::map<int, IntervalTracker> R;

public:
    PacketProcessor(int numBinsP, int numAddressesLog) 
        : numBins(numBinsP) , maxVal(1 << numAddressesLog)
    {
        histogram.resize(numBins, 0);
    }
        
    void processPacket(Packet& packet);
    void dumpHistogram();
};



