#include <iostream>

#include <chrono>
#include <thread>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"

#include "strategy.hpp"

std::string rebuy = "a $500";
std::string bet = "b $100";
std::string e = "e";

int main() {
    Game game;

    uint32_t stackSize = 0;

    Msg msg;

    // Play hands
    for (uint32_t hand = 0; hand < 10; hand++) {

        // Start the new hand, getting money if necessary 
        if (stackSize < 100) {
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
                if (stackSize < 100) {
                    msg = Strategy::executeAction(game, rebuy);
                    stackSize = msg.stackSize;
                }
            }
            
            msg = Strategy::executeAction(game, action);
            stackSize = msg.stackSize;

            if (iters++ > 6) {
                exit(0);
            }
        }
    }

    Strategy::executeAction(game, e);
}

