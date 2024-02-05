#include <string>
#include <iostream>
#include <chrono>
#include <thread>

#include "engine/msg.hpp"
#include "engine/game.hpp"

int delay = 0; // milliseconds

enum ACTION {
    S,
    H,
    D,
};

std::string toString(ACTION a) {
    switch (a) {
        case S: return "s";
        case H: return "h";
        case D: return "d";
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

// Strategy is called on an active board 
namespace Strategy {


    Msg executeAction(Game& game, std::string action) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));

        std::cout << "Executing Action: " << action << std::endl;
        Msg msg = game.processInput(action);
        msg.print();
        return msg;
    }


    std::string generateAction(Msg& msg) {
        Hand& hand = msg.playerHands[msg.playerIdx];
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

