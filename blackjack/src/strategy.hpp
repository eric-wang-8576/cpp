#include <string>
#include <iostream>
#include <chrono>
#include <thread>

#include "engine/msg.hpp"
#include "engine/game.hpp"

#define VERBOSE false

enum ACTION {
    S,
    H,
    D, // D/H
    O, // D/S
    P, 
};

enum SPLIT {
    Y,
    N,
};

inline std::string toString(ACTION a) {
    switch (a) {
        case S: return "s";
        case H: return "h";
        case D: return "d";
        case P: return "p";
        default: return "x";
    }
}

/*
 * First, pass through pair splitting matrix
 * Second, soft totals
 * Third, hard totals
 */

ACTION hardTotals[8][10] = {
    {S, S, S, S, S, H, H, H, H, H}, // 16
    {S, S, S, S, S, H, H, H, H, H}, // 15 
    {S, S, S, S, S, H, H, H, H, H}, // 14
    {S, S, S, S, S, H, H, H, H, H}, // 13
    {H, H, S, S, S, H, H, H, H, H}, // 12
    {D, D, D, D, D, D, D, D, D, D}, // 11
    {D, D, D, D, D, D, D, D, H, H}, // 10
    {H, D, D, D, D, H, H, H, H, H}, // 9
};

ACTION softTotals[7][10] = {
    {S, S, S, S, O, S, S, S, S, S}, // A, 8
    {O, O, O, O, O, S, S, H, H, H}, // A, 7
    {H, D, D, D, D, H, H, H, H, H}, // A, 6
    {H, H, D, D, D, H, H, H, H, H}, // A, 5
    {H, H, D, D, D, H, H, H, H, H}, // A, 4
    {H, H, H, D, D, H, H, H, H, H}, // A, 3
    {H, H, H, D, D, H, H, H, H, H}, // A, 2
};

SPLIT pairSplitting[8][10] = {
    {Y, Y, Y, Y, Y, N, Y, Y, N, N}, // 9
    {Y, Y, Y, Y, Y, Y, Y, Y, Y, Y}, // 8
    {Y, Y, Y, Y, Y, Y, N, N, N, N}, // 7
    {Y, Y, Y, Y, Y, N, N, N, N, N}, // 6
    {N, N, N, N, N, N, N, N, N, N}, // 5
    {N, N, N, Y, Y, N, N, N, N, N}, // 4
    {Y, Y, Y, Y, Y, Y, N, N, N, N}, // 3
    {Y, Y, Y, Y, Y, Y, N, N, N, N}, // 2
};

// Strategy is called on an active board 
namespace Strategy {

    void executeAction(Game& game, std::string action, Msg& msg, uint32_t delay) {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));

        if constexpr(VERBOSE) {
            std::cout << "Executing Action: " << action << std::endl;
        }
        game.processInput(action, msg);
        if constexpr(VERBOSE) {
            msg.print();
        }
        return;
    }

    std::string generateAction(Msg& msg) {
        // If it is a pair, check to see if we split
        Hand& hand = msg.playerHandsP[msg.playerIdx];

        uint8_t upCard = msg.dealerHandP->getFirstCardValue();

        SPLIT s = N;
        if (hand.isPair()) {
            uint8_t value = hand.getFirstCardValue();
            if (value == 11) {
                s = Y;
            } else if (value == 10) {
                s = N;
            } else {
                s = pairSplitting[9 - value][upCard - 2];
            }
        }
        
        if (s == Y) {
            return toString(P);
        }

        // If we have a soft hand, use the soft total table
        if (hand.isSoft() && hand.getPrimaryVal() < 20) {
            uint8_t smallerTotal = hand.getPrimaryVal() - 10;
            ACTION a = softTotals[9 - smallerTotal][upCard - 2];
            
            if (a == D && hand.getNumCards() > 2) {
                a = H;
            }

            if (a == O) {
                if (hand.getNumCards() == 2) {
                    a = D;
                } else {
                    a = S;
                }
            }

            return toString(a);
        }

        // Default to the hard totals table
        uint8_t total = hand.getPrimaryVal();
        if (total <= 8) {
            return toString(H);

        } else if (total >= 17) {
            return toString(S);

        } else {
            ACTION a = hardTotals[16 - total][upCard - 2];
            if (hand.getNumCards() != 2 && a == D) {
                a = H;
            }
            return toString(a);
        }
    }

}

