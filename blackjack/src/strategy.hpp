#include <string>
#include <iostream>
#include <chrono>
#include <thread>

#include "engine/msg.hpp"
#include "engine/game.hpp"

#define DELAY 0 // milliseconds

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

std::string toString(ACTION a) {
    switch (a) {
        case S: return "s";
        case H: return "h";
        case D: return "d";
        case P: return "p";
        default: return "X";
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
    {H, H, H, D, D, H, H, H, H, H}, // A, 3
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

#define VERBOSE false

// Strategy is called on an active board 
namespace Strategy {

    Msg executeAction(Game& game, std::string action) {
        std::this_thread::sleep_for(std::chrono::milliseconds(DELAY));

        if constexpr(VERBOSE) {
            std::cout << "Executing Action: " << action << std::endl;
        }
        Msg msg = game.processInput(action);
        if constexpr(VERBOSE) {
            msg.print();
        }
        return msg;
    }

    std::string generateAction(Msg& msg) {
        // If it is a pair, check to see if we split
        Hand& hand = msg.playerHands[msg.playerIdx];
        uint8_t upCard = msg.dealerHand.cards[0].getVal();

        SPLIT s;
        if (hand.cards.size() >= 2 && hand.isPair()) {
            uint8_t value = hand.cards[0].getVal();
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
        if (hand.cards.size() >= 2 && hand.isSoft()) {
            uint8_t smallerTotal = hand.values[0];
            ACTION a;
            if (smallerTotal == 10) {
                a = S;
            } else {
                a = softTotals[9 - smallerTotal][upCard - 2];
            }
            
            if (a == D && hand.cards.size()) {
                a = H;
            }

            if (a == O) {
                if (hand.cards.size() == 2) {
                    a = D;
                } else {
                    a = S;
                }
            }

            return toString(a);
        }

        // Default to the hard totals table
        uint8_t total = hand.values.back();
        if (total <= 8) {
            return toString(H);
        } else if (total >= 17) {
            return toString(S);
        } else {
            uint8_t upCard = msg.dealerHand.cards[0].getVal();
            ACTION a = hardTotals[16 - total][upCard - 2];
            if (hand.cards.size() != 2 && a == D) {
                a = H;
            }
            return toString(a);
        }
    }

}

