#include <iostream>

#include <chrono>
#include <thread>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"

#include "strategy.hpp"

#define NUMHANDS 10000000
#define MAXBETSIZE 10000

#define COUNTING 0

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
            Strategy::executeAction(game, rebuy, msg);
            stackSize = msg.stackSize;
        }

        Strategy::executeAction(game, smallBet, msg);
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
                std::cout << "FAIL!" << std::endl;
                std::cout << action << std::endl;
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
    std::cout << "# smallBets: " << smallBets << std::endl;
    std::cout << "# bigBets: " << bigBets << std::endl;
}

