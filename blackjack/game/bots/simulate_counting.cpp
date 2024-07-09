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
std::string smallBetSize = "b $2";
std::string bigBetSize = "b $10000";
std::string e = "e";

std::mutex mutex;

int getCount(uint8_t cardID) {
    switch (cardID) {
    case 1:
        return -4;
    case 2:
        return 2;
    case 3:
        return 3;
    case 4:
        return 3;
    case 5:
        return 4;
    case 6:
        return 3;
    case 7:
        return 2;
    case 8:
        return 0;
    case 9:
        return -1;
    case 10:
    case 11:
    case 12:
    case 13:
        return -3;
    default:
        std::cout << "Count Error, ID = " << (int) cardID << std::endl;
        exit(1);
    }
}

double calculateCount(int count, int numCardsRemaining) {
    return (double) count / ((double) numCardsRemaining / 52);
}


// Returns the PNL
void runSimulation(uint32_t numHands, uint32_t numTrialsPerThread, double threshold, std::string& PNLstr) {
    PNLstr.reserve(numTrialsPerThread * 10);
    for (int trial = 0; trial < numTrialsPerThread; ++trial) {
        int numDecks = 6;

        // Counting state
        int count = 0;
        int numCardsRemaining = numDecks * 52;
        double trueCount = calculateCount(count, numCardsRemaining);

        // Bot specific
        Game game {numDecks};

        uint32_t stackSize = 0;

        Msg msg;

        // Play hands
        for (uint32_t hand = 0; hand < numHands; hand++) {

            // Start the new hand, getting money if necessary 
            if (stackSize < MAXBETSIZE) {
                Strategy::executeAction(game, rebuy, msg, 0);
                stackSize = msg.stackSize;
            }

            // Decide whether or not to bet big based on the count
            // and whether or not we are about to start a new hand 
            int prevStackSize = stackSize;
            if (trueCount >= threshold && numCardsRemaining > 52 * numDecks / 3) {
                Strategy::executeAction(game, bigBetSize, msg, 0);
            } else {
                Strategy::executeAction(game, smallBetSize, msg, 0);
            }
            stackSize = msg.stackSize;

            // If the deck was shuffled reset all the values
            if (msg.shuffled) {
                count = 0;
                numCardsRemaining = numDecks * 52;
                trueCount = calculateCount(count, numCardsRemaining);
            }

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

            // Update count
            for (int idx = 0; idx < msg.dealerHandP->getNumCards(); ++idx) {
                count += getCount(msg.dealerHandP->getCardID(idx));
                numCardsRemaining--;
            }
            for (int playerIdx = 0; playerIdx < msg.numPlayerHands; ++playerIdx) {
                 Hand& hand = msg.playerHandsP[playerIdx];
                for (int idx = 0; idx < hand.getNumCards(); ++idx) {
                    count += getCount(hand.getCardID(idx));
                    numCardsRemaining--;
                }
            }
            trueCount = calculateCount(count, numCardsRemaining);
        }

        game.processInput("e", msg);
        PNLstr += std::to_string(msg.PNL) + "\n";
    }
}


int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cout << "Please specify number of hands, number of trials, and counting threshold" << std::endl;
        exit(1);
    }

    uint32_t numHands = std::stoi(argv[1]);
    uint32_t numTrials = std::stoi(argv[2]);
    double threshold = std::stoi(argv[3]);

    // Launch Threads
    std::vector<std::string> PNLs(NUMTHREADS);
    std::vector<std::thread> threads;
    for (int i = 0; i < NUMTHREADS; ++i) {
        threads.emplace_back(runSimulation, numHands, numTrials / NUMTHREADS, threshold, std::ref(PNLs[i]));
    }

    for (auto& th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    std::cout << "PNL Distribution for " + valueToString(numHands) + " hands with counting -> " + valueToString(numTrials) + " trials" << std::endl;
    for (int i = 0; i < NUMTHREADS; ++i) {
        std::cout << PNLs[i];
    }
}

