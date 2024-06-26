#include <iostream>

#include <chrono>
#include <thread>

#include "game.hpp"
#include "strategy.hpp"

#define NUMTRIALS 10000

int main(int numArgs, char* argv[]) {

    if (numArgs != 5) {
        std::cout << "Please specify starting stack, bet size, lower bound, and upper bound" << std::endl;
        exit(1);
    }

    int startingStack = std::atoi(argv[1]);
    int betSize = std::atoi(argv[2]);
    int lower = std::atoi(argv[3]);
    int upper = std::atoi(argv[4]);

    uint32_t numWin = 0;
    uint32_t numLoss = 0;

    std::string rebuyStr = "a $" + std::to_string(betSize);
    std::string betStr = "b $" + std::to_string(betSize);

    Msg msg;

    // Play hands
    for (uint32_t trial = 0; trial < NUMTRIALS; trial++) {
        Game game {6};

        // Initialize
        Strategy::executeAction(game, "a $" + std::to_string(startingStack), msg, 0);
        int stackSize = msg.stackSize;

        while (true) {
            // If we need more money for this action, get it
            if (stackSize < betSize) {
                Strategy::executeAction(game, rebuyStr, msg, 0);
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
                        stackSize = msg.stackSize;
                    }
                }
                
                Strategy::executeAction(game, action, msg, 0);
                stackSize = msg.stackSize;
            }

            if (stackSize <= lower) {
                numLoss++;
                break;
            } else if (stackSize >= upper) {
                numWin++;
                break;
            }
        }


        game.processInput("e", msg);
        msg.print();
    }

    std::cout << "# of times ended at $" + std::to_string(upper) << ": " << numWin << std::endl;
    std::cout << "# of times ended at $" + std::to_string(lower) << ": " << numLoss << std::endl;
    std::cout << std::endl;
    std::cout << "Percentage Wins: " << (100 * (float) numWin)/((float) NUMTRIALS) 
              << "%" << std::endl;
    std::cout << std::endl;

}

