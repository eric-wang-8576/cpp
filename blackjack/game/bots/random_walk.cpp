#include <iostream>

#include <chrono>
#include <thread>

#include "game.hpp"
#include "strategy.hpp"

#define NUMTRIALS 16000
#define NUMTHREADS 32

void runSimulation(
    uint32_t numTrials, 
    int startingStack,
    int betSize, 
    int lowerBound, 
    int upperBound,
    int& numWinP, 
    int& numLossP
) {
    std::string rebuyStr = "a $" + std::to_string(betSize);
    std::string betStr = "b $" + std::to_string(betSize);

    int numWin = 0;
    int numLoss = 0;

    Msg msg;

    // Play hands
    for (uint32_t trial = 0; trial < numTrials; trial++) {
        Game game {6};

        // Initialize
        int buyIn = startingStack;
        int stackSize = startingStack;

        Strategy::executeAction(game, "a $" + std::to_string(startingStack), msg, 0);

        while (true) {
            // If we need more money for this action, get it
            if (stackSize < betSize) {
                Strategy::executeAction(game, rebuyStr, msg, 0);
                buyIn += betSize;
                stackSize = msg.stackSize;
            }

            Strategy::executeAction(game, betStr, msg, 0);
            stackSize = msg.stackSize;

            // While we are in the hand, request actions
            while (msg.playerIdx != msg.numPlayerHands) {
                std::string action = Strategy::generateAction(msg);

                // If we need more money for this action, get it
                if (action == "d" || action == "p") {
                    if (stackSize < betSize) {
                        Strategy::executeAction(game, rebuyStr, msg, 0);
                        buyIn += betSize;
                        stackSize = msg.stackSize;
                    }
                }
                
                Strategy::executeAction(game, action, msg, 0);
                stackSize = msg.stackSize;
            }

            int PNL = stackSize - buyIn;
            if (PNL <= lowerBound) {
                numLoss++;
                break;
            } else if (PNL >= upperBound) {
                numWin++;
                break;
            }
        }

        game.processInput("e", msg);
    }

    numWinP = numWin;
    numLossP = numLoss;
}


int main(int numArgs, char* argv[]) {

    if (numArgs != 5) {
        std::cout << "Please specify starting stack, bet size, lower bound, and upper bound" << std::endl;
        exit(1);
    }

    int startingStack = std::atoi(argv[1]);
    int betSize = std::atoi(argv[2]);
    int lower = std::atoi(argv[3]);
    int upper = std::atoi(argv[4]);

    int lowerBound = lower - startingStack;
    int upperBound = upper - startingStack;

    int numTrialsPerThread = NUMTRIALS / NUMTHREADS;

    std::vector<std::thread> threads;
    std::vector<int> numWin(NUMTHREADS);
    std::vector<int> numLoss(NUMTHREADS);

    for (int i = 0; i < NUMTHREADS; ++i) {
        threads.emplace_back(
            runSimulation,
            numTrialsPerThread,
            startingStack,
            betSize,
            lowerBound,
            upperBound,
            std::ref(numWin[i]), 
            std::ref(numLoss[i]));
    }

    for (auto& th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    int totalWin = std::accumulate(numWin.begin(), numWin.end(), 0);
    int totalLoss = std::accumulate(numLoss.begin(), numLoss.end(), 0);

    std::cout << "# of times ended at " + priceToString(upper) << ": " << totalWin << std::endl;
    std::cout << "# of times ended at " + priceToString(lower) << ": " << totalLoss << std::endl;
    std::cout << std::endl;
    std::cout << "Percentage Wins: " << (100 * (float) totalWin)/((float) NUMTRIALS) 
              << "%" << std::endl;
    std::cout << std::endl;

}

