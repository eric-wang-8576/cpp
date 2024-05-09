#include <iostream>
#include "packet_generator.hpp"
#include "packet_processor.hpp"

#define NUMPACKETS 10000
#define NUMBINS 16

int main() {
    int numAddressesLog = 5;
    int numTrainsLog = 4;
    double meanTrainSize = 5;
    double meanTrainsPerComm = 4;
    int meanWindow = 5;
    int meanCommsPerAddress = 3;
    int meanWork = 3000;
    double configFraction = 0.1;
    double pngFraction = 0.2;
    double acceptingFraction = 0.8;

    PacketGenerator gen {
        numAddressesLog,
        numTrainsLog,
        meanTrainSize,
        meanTrainsPerComm,
        meanWindow,
        meanCommsPerAddress,
        meanWork,
        configFraction,
        pngFraction,
        acceptingFraction
    };

    PacketProcessor p(NUMBINS, numAddressesLog);
    for (int i = 0; i < NUMPACKETS; i++) {
        Packet pkt = gen.getPacket();
        pkt.printPacket();
        p.processPacket(pkt);
    }
    p.dumpHistogram();
}
