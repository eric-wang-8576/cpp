#include <iostream>
#include "packet_generator.hpp"
#include "packet_processor.hpp"
#include "parallel_packet_processor.hpp"

#define NUMLOCKS 1024

#define NUMBATCHES 300
#define NUMBINS 16
#define PACKETBATCHSIZE 1000

#define NUMPARAMMIXES 8
#define NUMTHREADTRIALS 6

int numThreads[NUMTHREADTRIALS] = {1, 2, 4, 8, 16, 32};
int numAddressesLog[NUMPARAMMIXES] = {11, 12, 12, 14, 15, 15, 15, 16};
int numTrainsLog[NUMPARAMMIXES] = {12, 10, 10, 10, 14, 15, 15, 14};
double meanTrainSize[NUMPARAMMIXES] = {5, 1, 4, 5, 9, 9, 10, 15};
double meanTrainsPerComm[NUMPARAMMIXES] = {1, 3, 3, 5, 16, 10, 13, 12};
int meanWindow[NUMPARAMMIXES] = {3, 3, 6, 6, 7, 9, 8, 9};
int meanCommsPerAddress[NUMPARAMMIXES] = {3, 1, 2, 2, 10, 9, 10, 5};
int meanWork[NUMPARAMMIXES] = {3822, 2644, 1304, 315, 4007, 7125, 5328, 8840};
double configFraction[NUMPARAMMIXES] = {0.24, 0.11, 0.10, 0.08, 0.02, 0.01, 0.04, 0.04};
double pngFraction[NUMPARAMMIXES] = {0.04, 0.09, 0.03, 0.05, 0.10, 0.20, 0.18, 0.19};
double acceptingFraction[NUMPARAMMIXES] = {0.96, 0.92, 0.90, 0.90, 0.84, 0.77, 0.80, 0.87};

int main() {
    for (int t = 0; t < NUMTHREADTRIALS; ++t) {
        for (int param = 0; param < NUMPARAMMIXES; ++param) {
            PacketGenerator gen {
                numAddressesLog[param],
                numTrainsLog[param],
                meanTrainSize[param],
                meanTrainsPerComm[param],
                meanWindow[param],
                meanCommsPerAddress[param],
                meanWork[param],
                configFraction[param],
                pngFraction[param],
                acceptingFraction[param]
            };

            ParallelPacketProcessor p(NUMBINS, numAddressesLog[param], NUMLOCKS, numThreads[t]);

            
            // packets to get permission tables in steady state
            int numInitPackets = (int) pow((double) (1 << numAddressesLog[param]), 1.5);
            for (int i = 0; i < numInitPackets; ++i) {
                p.processPacket(gen.getPacket());
            }

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < NUMBATCHES; i++) {
                std::vector<Packet> batch;
                for (int i = 0; i < PACKETBATCHSIZE; ++i) {
                    batch.push_back(gen.getPacket());
                }

                p.processPacketBatch(batch);
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = end - start;
            std::cout << "Parameter Mix #" << param + 1 << " with " << numThreads[t] << " threads took " << 
                dur.count() - 1000 << " milliseconds." << std::endl;
        }
    }
}
