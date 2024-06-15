#include <iostream>

#include <chrono>
#include <thread>
#include <numeric>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"

#include "strategy.hpp"

#define MAXBETSIZE 10000
#define NUMTHREADS 25
#define BETSIZE 2

std::string rebuy = "a $10000";
std::string betSize = "b $2";
std::string e = "e";

// Returns the PNL
void runSimulation(int numDecks, uint32_t numHands, int& PNL) {
    Game game {numDecks};

    uint32_t stackSize = 0;

    Msg msg;

    // Play hands
    for (uint32_t hand = 0; hand < numHands; hand++) {

        // Start the new hand, getting money if necessary 
        if (stackSize < MAXBETSIZE) {
            Strategy::executeAction(game, rebuy, msg);
            stackSize = msg.stackSize;
        }

        Strategy::executeAction(game, betSize, msg);
        stackSize = msg.stackSize;

        int iters = 0;
        // While we are in the hand, request actions
        while (msg.playerIdx != msg.numPlayerHands) {
            std::string action = Strategy::generateAction(msg);

            // If we need more money for this action, get it
            if (action == "h" || action == "d" || action == "p") {
                if (stackSize < MAXBETSIZE) {
                    Strategy::executeAction(game, rebuy, msg);
                    stackSize = msg.stackSize;
                }
            }
            
            Strategy::executeAction(game, action, msg);
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

    if (numArgs != 3) {
        std::cout << "Please specify the number of decks and hands" << std::endl;
        exit(1);
    }

    int numDecks = std::atoi(argv[1]);
    int numHands = std::atoi(argv[2]);

    // Launch Threads
    int numHandsPerThread = numHands / NUMTHREADS;

    std::vector<int> PNLs(NUMTHREADS);
    std::vector<std::thread> threads;
    for (int i = 0; i < NUMTHREADS; ++i) {
        threads.emplace_back(runSimulation, numDecks, numHandsPerThread, std::ref(PNLs[i]));
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
    std::cout << "Percentage PNL is " << ((float) totalPNL) / (numHands * 2) * 100 << "%" << std::endl;
    std::cout << "------------------------------\n" << std::endl;
    std::cout << "Total hands played is " << numHands << std::endl;
    std::cout << "Total PNL is " << totalPNL << std::endl;
    std::cout << "Avg ns per hand: " << duration.count() / numHands << std::endl;

}

