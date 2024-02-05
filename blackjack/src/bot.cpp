#include <iostream>

#include <chrono>
#include <thread>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"

#include "strategy.hpp"

#define NUMHANDS 1000000
#define BETSIZE 2
#define VERBOSE false

std::string rebuy = "a $100";
std::string bet = "b $2";
std::string e = "e";

int main() {
    Game game;

    uint32_t stackSize = 0;

    Msg msg;

    // Play hands
    for (uint32_t hand = 0; hand < NUMHANDS; hand++) {

        // Start the new hand, getting money if necessary 
        if (stackSize < BETSIZE) {
            msg = Strategy::executeAction(game, rebuy);
            stackSize = msg.stackSize;
        }

        // Bet $1
        msg = Strategy::executeAction(game, bet);
        stackSize = msg.stackSize;

        int iters = 0;
        // While we are in the hand, request actions
        while (msg.playerIdx != msg.playerHands.size()) {
            std::string action = Strategy::generateAction(msg);

            // If we need more money for this action, get it
            if (action == "h" || action == "d" || action == "p") {
                if (stackSize < BETSIZE) {
                    msg = Strategy::executeAction(game, rebuy);
                    stackSize = msg.stackSize;
                }
            }
            
            msg = Strategy::executeAction(game, action);
            stackSize = msg.stackSize;

            if (iters++ > 100) {
                exit(0);
            }
        }
    }

    msg = game.processInput("e");
    msg.print();
}

