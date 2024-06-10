#include <iostream>

#include <chrono>
#include <thread>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"

#include "strategy.hpp"

#define NUMHANDS 1000000
#define MAXBETSIZE 10000

std::string rebuy = "a $10000";
std::string smallBet = "b $2";
std::string bigBet = "b $10000";
std::string e = "e";

int main(int numArgs, char** argv) {
    if (numArgs != 2) {
        std::cout << "Please specify the number of decks" << std::endl;
        exit(1);
    }

    // TODO: Improve this code
    int numDecks = (int) (*argv[1] - '0');
    Game game {numDecks};

    uint32_t stackSize = 0;

    uint32_t smallBets = 0;
    uint32_t bigBets = 0;

    Msg msg;

    // Play hands
    for (uint32_t hand = 0; hand < NUMHANDS; hand++) {

        // Start the new hand, getting money if necessary 
        if (stackSize < MAXBETSIZE) {
            msg = Strategy::executeAction(game, rebuy);
            stackSize = msg.stackSize;
        }

        // Bet big if count is good
        if (msg.count / numDecks >= 2) {
            msg = Strategy::executeAction(game, bigBet);
            bigBets++;
        } else {
            msg = Strategy::executeAction(game, smallBet);
            smallBets++;
        }
        stackSize = msg.stackSize;

        int iters = 0;
        // While we are in the hand, request actions
        while (msg.playerIdx != msg.playerHands.size()) {
            std::string action = Strategy::generateAction(msg);

            // If we need more money for this action, get it
            if (action == "h" || action == "d" || action == "p") {
                if (stackSize < MAXBETSIZE) {
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
    std::cout << "# smallBets: " << smallBets << std::endl;
    std::cout << "# bigBets: " << bigBets << std::endl;
}

