#include <iostream>

#include <chrono>
#include <thread>
#include <numeric>
#include <iostream>
#include <iomanip>

#include "engine/game.hpp"
#include "engine/shoe.hpp"
#include "engine/hand.hpp"

#include "strategy.hpp"

#define MAXBETSIZE 10000
#define BETSIZE 2

std::string rebuy = "a $10000";
std::string smallBetSize = "b $2";
std::string bigBetSize = "b $10000";
std::string e = "e";

int getCount(uint8_t cardID) {
    if (2 <= cardID && cardID <= 6) {
        return 1;
    } else if (7 <= cardID && cardID <= 9) {
        return 0;
    } else if (10 <= cardID && cardID <= 13 || cardID == 1) {
        return -1;
    } else {
        std::cout << "Count Error, ID = " << (int) cardID << std::endl;
        exit(1);
    }
}

struct comma_numpunct : std::numpunct<char> {
protected:
    virtual char do_thousands_sep() const {
        return ',';
    }

    virtual std::string do_grouping() const {
        return "\3";
    }
};

double calculateCount(int count, int numCardsRemaining) {
    return (double) count / ((double) numCardsRemaining / 52);
}

// Returns the PNL
void runSimulation(int numDecks, uint32_t numHands, int64_t& PNL, uint32_t& amountBet, int& numSmallBets, int& numBigBets, uint32_t delay, double threshold) {
    // Counting state
    int count = 0;
    int numCardsRemaining = numDecks * 52;
    double trueCount = calculateCount(count, numCardsRemaining);

    // Bot specific
    Game game {numDecks};

    uint32_t stackSize = 0;

    amountBet = 0;
    numSmallBets = 0;
    numBigBets = 0;

    Msg msg;

    // Play hands
    for (uint32_t hand = 0; hand < numHands; hand++) {

        // Start the new hand, getting money if necessary 
        if (stackSize < MAXBETSIZE) {
            Strategy::executeAction(game, rebuy, msg, delay);
            stackSize = msg.stackSize;
        }

        // Decide whether or not to bet big based on the count
        // and whether or not we are about to start a new hand 
        if (trueCount >= threshold && numCardsRemaining > 52 * numDecks / 3 + 1) {
            amountBet += 10000;
            numBigBets++;
            Strategy::executeAction(game, bigBetSize, msg, delay);
        } else {
            amountBet += 2;
            numSmallBets++;
            Strategy::executeAction(game, smallBetSize, msg, delay);
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
    msg.print();

    PNL = msg.PNL;
}


int main(int numArgs, char* argv[]) {
    std::locale comma_locale(std::locale(), new comma_numpunct());
    std::cout.imbue(comma_locale);

    auto start = std::chrono::high_resolution_clock::now();

    if (numArgs != 6) {
        std::cout << "Please specify the number of decks, hands, threads, delay in ms, and counting threshold" << std::endl;
        exit(1);
    }

    int numDecks = std::atoi(argv[1]);
    int numHands = std::atoi(argv[2]);
    int numThreads = std::atoi(argv[3]);
    int delay = std::atoi(argv[4]);
    double threshold = std::atof(argv[5]);

    // Launch Threads
    int numHandsPerThread = numHands / numThreads;

    std::vector<int64_t> PNLs(numThreads);
    std::vector<uint32_t> amounts(numThreads);
    std::vector<int> numSmallBets(numThreads);
    std::vector<int> numBigBets(numThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(runSimulation, numDecks, numHandsPerThread, 
                             std::ref(PNLs[i]), std::ref(amounts[i]), 
                             std::ref(numSmallBets[i]), std::ref(numBigBets[i]), delay, threshold);
    }

    for (auto& th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "\n\n\n\n\n" << std::endl;
    int64_t totalPNL = std::accumulate(PNLs.begin(), PNLs.end(), 0);
    uint32_t amountBet = std::accumulate(amounts.begin(), amounts.end(), 0);
    int numSmall = std::accumulate(numSmallBets.begin(), numSmallBets.end(), 0);
    int numBig = std::accumulate(numBigBets.begin(), numBigBets.end(), 0);
    std::cout << "------------------------------" << std::endl;
    std::cout << "PNL per hand is $" << std::fixed << std::setprecision(2) << float(totalPNL) / float(numHands) << std::endl;
    std::cout << "------------------------------\n" << std::endl;
    std::cout << "Percentage PNL is " << float(totalPNL) / float(amountBet) * 100 << "%" << std::endl;
    std::cout << "Total hands played is " << numHands << std::endl;
    std::cout << "# of small bets: " << numSmall << std::endl;
    std::cout << "# of big bets: " << numBig << std::endl;
    std::cout << "Total PNL is " << (totalPNL < 0 ? "-" : "") << "$" << (totalPNL > 0 ? totalPNL : -totalPNL) << std::endl;
    std::cout << "Avg ns per hand: " << duration.count() / numHands << std::endl;
    std::cout << "" << std::endl;

}

