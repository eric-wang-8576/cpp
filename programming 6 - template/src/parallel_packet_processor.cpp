#include "parallel_packet_processor.hpp"

void ParallelPacketProcessor::processPacketBatch(std::vector<Packet>& batch) {
    tp.enqueue([this] (std::vector<Packet>& batch) 
            { 

                for (Packet& packet : batch) {
                    this->pp.processPacket(packet); 
                }

            }, 
            batch);
}
    
void ParallelPacketProcessor::processPacket(const Packet& packet) {
    tp.enqueue([this] (Packet& x) { this->pp.processPacket(x); }, packet);
}

void ParallelPacketProcessor::dumpHistogram() {
    pp.dumpHistogram();
}
