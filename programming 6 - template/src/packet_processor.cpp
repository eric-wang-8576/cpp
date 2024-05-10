#include "packet_processor.hpp"

void PacketProcessor::processPacket(const Packet& packet) {
    if (packet.type == MessageType::ConfigPacket) {
        const Config& config = packet.config;

        // Modify PNG
        if (config.personaNonGrata) {
            PNG.add(config.address);
        } else {
            PNG.remove(config.address);
        }

        // Modify R
        {
            if (config.address < 0 || config.address >= maxAddr) {
                return;
            }
            R[config.address]->setRange(config.addressBegin, config.addressEnd - 1, config.acceptingRange);
        }

    } else {
        const Header& header = packet.header;

        // Check that PNG[S] = true
        if (!PNG.contains(header.source)) {
            return;
        }

        // Check that S is in R[dest]
        {
            if (header.dest < 0 || header.dest >= maxAddr) {
                return;
            }
            if (!(R[header.dest]->contains(header.source))) {
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
