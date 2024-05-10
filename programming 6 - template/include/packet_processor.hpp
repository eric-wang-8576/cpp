#pragma once
#include "fingerprint.hpp"
#include "packet_generator.hpp"
#include "interval_tracker.hpp"
#include "hashset.hpp"
#include <set>
#include <mutex>


class PacketProcessor {
    int numBins;
    int maxAddr;
    std::vector<int> histogram;
    FingerPrint f;

    HashSet PNG;
    std::vector<std::unique_ptr<IntervalTracker>> R;
    int numLocks;

public:
    PacketProcessor(int numBinsP, int numAddressesLog, int numLocksP) 
        : numBins(numBinsP), maxAddr(1 << numAddressesLog), numLocks(numLocksP),
          PNG(numLocksP, 4)
    {
        histogram.resize(numBins, 0);
        R.reserve(maxAddr);
        for (int i = 0; i < maxAddr; ++i) {
            R.emplace_back(std::make_unique<IntervalTracker>(0, maxAddr));
        }
    }
        
    void processPacket(const Packet& packet);
    void dumpHistogram();
};



