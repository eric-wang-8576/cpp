#include <iostream>

#include <chrono>
#include <thread>
#include <numeric>

#include "game.hpp"
#include "strategy.hpp"

#define MAXBETSIZE 10000
#define BETSIZE 2

std::string rebuy = "a $10000";
std::string betSize = "b $10";
std::string e = "e";

// Returns the PNL
void runSimulation(int numDecks, uint32_t numHands, int& PNL, uint32_t delay) {
    Game game {numDecks};

    uint32_t stackSize = 0;

    Msg msg;

    // Play hands
    for (uint32_t hand = 0; hand < numHands; hand++) {

        // Start the new hand, getting money if necessary 
        if (stackSize < MAXBETSIZE) {
            Strategy::executeAction(game, rebuy, msg, delay);
            stackSize = msg.stackSize;
        }

        Strategy::executeAction(game, betSize, msg, delay);
        stackSize = msg.stackSize;

        int iters = 0;
        // While we are in the hand, request actions
        while (msg.playerIdx != msg.numPlayerHands) {
            std::string action = Strategy::generateAction(msg);

            // If we need more money for this action, get it
            if (action == "h" || action == "d" || action == "p") {
                if (stackSize < MAXBETSIZE) {
                    Strategy::executeAction(game, rebuy, msg, delay);
                    stackSize = msg.stackSize;
                }
            }
            
            Strategy::executeAction(game, action, msg, delay);
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
    msg.print();

    PNL = msg.PNL;
}


int main(int numArgs, char* argv[]) {

    auto start = std::chrono::high_resolution_clock::now();

    if (numArgs != 5) {
        std::cout << "Please specify the number of decks, hands, threads, and delay in ms" << std::endl;
        exit(1);
    }

    int numDecks = std::atoi(argv[1]);
    int numHands = std::atoi(argv[2]);
    int numThreads = std::atoi(argv[3]);
    int delay = std::atoi(argv[4]);

    // Launch Threads
    int numHandsPerThread = numHands / numThreads;

    std::vector<int> PNLs(numThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(runSimulation, numDecks, numHandsPerThread, std::ref(PNLs[i]), delay);
    }

    for (auto& th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "\n\n\n\n\n" << std::endl;
    int totalPNL = std::accumulate(PNLs.begin(), PNLs.end(), 0);
    std::cout << "------------------------------" << std::endl;
    std::cout << "Percentage PNL is " << ((float) totalPNL) / (numHands * 10) * 100 << "%" << std::endl;
    std::cout << "------------------------------\n" << std::endl;
    std::cout << "Total hands played is " << numHands << std::endl;
    std::cout << "Total PNL is " << totalPNL << std::endl;
    std::cout << "Avg ns per hand: " << duration.count() / numHands << std::endl;
    std::cout << "" << std::endl;

}

