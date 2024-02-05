#include <iostream>

#include <chrono>
#include <thread>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"

#include "strategy.hpp"

#define NUMTRIALS 1000000

int main() {

    uint32_t numWin = 0;
    uint32_t numLoss = 0;


    Msg msg;

    // Play hands
    for (uint32_t trial = 0; trial < NUMTRIALS; trial++) {
        Game game;

        // Start with $500 
        uint32_t buyIn = 500;
        uint32_t stackSize = 500;
        msg = Strategy::executeAction(game, "a $500");

        while (true) {
            // If we need more money for this action, get it
            if (stackSize < 100) {
                msg = Strategy::executeAction(game, "a $100");
                buyIn += 100;
                stackSize = msg.stackSize;
            }

            msg = Strategy::executeAction(game, "b $100");
            stackSize = msg.stackSize;

            // While we are in the hand, request actions
            while (msg.playerIdx != msg.playerHands.size()) {
                std::string action = Strategy::generateAction(msg);

                // If we need more money for this action, get it
                if (action == "d" || action == "p") {
                    if (stackSize < 100) {
                        msg = Strategy::executeAction(game, "a $100");
                        buyIn += 100;
                        stackSize = msg.stackSize;
                    }
                }
                
                msg = Strategy::executeAction(game, action);
                stackSize = msg.stackSize;
            }

            int32_t PNL = (int32_t) stackSize - (int32_t) buyIn;
            if (PNL <= -500) {
                numLoss++;
                break;
            } else if (PNL >= 500) {
                numWin++;
                break;
            }
        }


        msg = game.processInput("e");
//        msg.print();
    }

    std::cout << "# of times made it to $1,000: " << numWin << std::endl;
    std::cout << "# of times lost the $500: " << numLoss << std::endl;
    std::cout << "Percentage Wins: " << (100 * (float) numWin)/((float) NUMTRIALS) 
              << "%" << std::endl;

}

