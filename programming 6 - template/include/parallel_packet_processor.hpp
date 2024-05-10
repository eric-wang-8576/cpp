#include "packet_processor.hpp"
#include "thread_pool.hpp"

class ParallelPacketProcessor {
    ThreadPool tp;
    PacketProcessor pp;

public:
    ParallelPacketProcessor(int numBinsP, int numAddressesLog, int numLocksP, int numThreadsP)
        : tp(numThreadsP), pp(numBinsP, numAddressesLog, numLocksP) {}

    void processPacketBatch(std::vector<Packet>& batch);
    void processPacket(const Packet& packet);
    void dumpHistogram();
};

