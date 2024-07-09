#include <iostream>

#include <chrono>
#include <thread>
#include <numeric>
#include <sstream>

#include "game.hpp"
#include "strategy.hpp"

#define MAXBETSIZE 10000
#define NUMTRIALS 10000
#define NUMTHREADS 20

std::string rebuy = "a $10000";
std::string e = "e";

std::mutex mutex;

// Returns the PNL
void runSimulation(uint32_t numHands, uint32_t numTrialsPerThread, const std::string& betSize, std::string& PNLstr) {
    PNLstr.reserve(numTrialsPerThread * 10);
    for (int trial = 0; trial < numTrialsPerThread; ++trial) {
        Game game {6};

        uint32_t stackSize = 0;

        Msg msg;

        // Play hands
        for (uint32_t hand = 0; hand < numHands; hand++) {

            // Start the new hand, getting money if necessary 
            if (stackSize < MAXBETSIZE) {
                Strategy::executeAction(game, rebuy, msg, 0);
                stackSize = msg.stackSize;
            }

            Strategy::executeAction(game, betSize, msg, 0);
            stackSize = msg.stackSize;

            int iters = 0;
            // While we are in the hand, request actions
            while (msg.playerIdx != msg.numPlayerHands) {
                std::string action = Strategy::generateAction(msg);

                // If we need more money for this action, get it
                if (action == "h" || action == "d" || action == "p") {
                    if (stackSize < MAXBETSIZE) {
                        Strategy::executeAction(game, rebuy, msg, 0);
                        stackSize = msg.stackSize;
                    }
                }
                
                Strategy::executeAction(game, action, msg, 0);
                stackSize = msg.stackSize;

                if (msg.prevActionConfirmation == "Invalid Action -> Please Try Again") {
                    exit(0);
                }

                if (iters++ > 10000) {
                    std::cout << "Iterations exceeded 10000" << std::endl;
                    std::cout << action << std::endl;
                    exit(0);
                }
            }
        }

        game.processInput("e", msg);
        
        PNLstr += std::to_string(msg.PNL) + "\n";
    }
}


int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cout << "Please specify bet string, number of hands, and number of trials" << std::endl;
        exit(1);
    }

    std::string betString = argv[1];
    uint32_t numHands = std::stoi(argv[2]);
    uint32_t numTrials = std::stoi(argv[3]);

    // Launch Threads
    std::vector<std::string> PNLs(NUMTHREADS);
    std::vector<std::thread> threads;
    for (int i = 0; i < NUMTHREADS; ++i) {
        threads.emplace_back(runSimulation, numHands, numTrials / NUMTHREADS, betString, std::ref(PNLs[i]));
    }

    for (auto& th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    std::cout << "PNL Distribution for " + valueToString(numHands) + " hands @ \'" + betString + "\' -> " + valueToString(numTrials) + " trials" << std::endl;
    for (int i = 0; i < NUMTHREADS; ++i) {
        std::cout << PNLs[i];
    }
}

