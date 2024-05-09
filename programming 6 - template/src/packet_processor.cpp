#include "packet_processor.hpp"

void PacketProcessor::processPacket(Packet& packet) {
    if (packet.type == MessageType::ConfigPacket) {
        Config& config = packet.config;

        // Modify PNG
        {
            std::unique_lock<std::mutex> lock(PNGlocks[config.address % numLocks]);
            PNG[config.address] = config.personaNonGrata;
        }

        // Modify R
        {
            std::unique_lock<std::mutex> lock(Rlocks[config.address % numLocks]);
            if (R.find(config.address) == R.end()) {
                R[config.address] = IntervalTracker(0, maxVal);
            }
            R[config.address].setRange(config.addressBegin, config.addressEnd - 1, config.acceptingRange);
        }

    } else {
        Header& header = packet.header;

        // Check that PNG[S] = true
        {
            std::unique_lock<std::mutex> lock(PNGlocks[config.address % numLocks]);
            if (PNG.find(header.source) == PNG.end()) {
                return;
            }
            if (PNG[header.source] == false) {
                return;
            }
        }

        // Check that S is in R[dest]
        {
            std::unique_lock<std::mutex> lock(Rlocks[config.address % numLocks]);
            if (R.find(header.dest) == R.end()) {
                return;
            }
            if (!(R[header.dest].contains(header.source))) {
                return;
            }
        }

        histogram[((int) f.getFingerprint(packet.body.iterations, packet.body.seed)) % numBins]++;
    }
}


void PacketProcessor::dumpHistogram() {
    std::cout << "[";
    for (int i = 0; i < numBins; ++i) {
        std::cout << histogram[i];
        if (i != numBins - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
