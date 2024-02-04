#include "game.hpp"

std::regex addPattern("a\\s\\$\\d+");

std::regex betPattern("b\\s\\$\\d+");
std::regex betPattern2("b");

void Game::resetBoard() {
    Hand hand;
    dealerHand = hand;

    playerHands.clear();
}

Msg Game::processInput(std::string input) {
    Msg msg;

    // Adding Chips
    if (std::regex_match(input, addPattern)) {
        
        uint32_t addValue = std::stoi(input.substr(3));
        buyIn += addValue;
        stackSize += addValue;

        msg.prevActionConfirmation = "You have added $" + 
                                     std::to_string(addValue) + 
                                     " to your stack. Your new stack size is $" +
                                     std::to_string(stackSize) + 
                                     ".";

        msg.stackSize = stackSize;
        msg.showBoard = activeBoard;
        msg.dealerHand = dealerHand;
        msg.playerHands = playerHands;
        msg.prompt = false;
        return msg;

    // Attempting to start a round 
    } else if (std::regex_match(input, betPattern) || 
               std::regex_match(input, betPattern2) ) {

        if (activeBoard) {
            goto invalidLabel;
        }

        // Set & Process betAmt
        uint32_t betAmt;
        if (input.length() == 1) {
            betAmt = prevBet;
        } else {
            betAmt = std::stoi(input.substr(3));
            prevBet = betAmt;
        }

        if (betAmt > stackSize) {
            msg.prevActionConfirmation = "Your current stack size is $" +
                                         std::to_string(stackSize) + 
                                         ". Please enter a smaller bet size or \
                                         add more money.";
            return msg;
        }


        // Formally starting round
        activeBoard = true;
        msg.prevActionConfirmation = "You have bet $" + std::to_string(betAmt) +
                                     ". The board is displayed below.";

        stackSize -= betAmt;

        // Give the dealer a hand, obscuring the second card
        dealerHand.addCard(shoe.draw());
        Card second = shoe.draw();
        second.setObscured(true);
        dealerHand.addCard(second);

        // Give the player a hand 
        Hand newHand;
        newHand.addCard(shoe.draw());
        newHand.addCard(shoe.draw());
        newHand.betAmt = betAmt;
        playerHands.push_back(newHand);

        // Check for blackjack TODO 

        // Send hands 
        msg.stackSize = stackSize;

        msg.showBoard = activeBoard;
        msg.dealerHand = dealerHand;
        msg.playerHands = playerHands;
        
        msg.prompt = true;
        msg.actionPrompt = "Option: hit or stand";
        return msg;

    } else if (input == "h") {
        
        if (!activeBoard) {
            goto invalidLabel;
        }
        Card card = shoe.draw();
        playerHands[0].addCard(card);

        // If we bust, then let the player know
        if (playerHands[0].busted) {
            activeBoard = false;

            msg.prevActionConfirmation = "You bust!";
            msg.stackSize = stackSize;

            msg.showBoard = true;
            msg.dealerHand = dealerHand;
            msg.playerHands = playerHands;
            resetBoard();

            msg.prompt = true;
            msg.actionPrompt = "Option: bet";

            return msg;

        // If no bust, then continue prompting the player
        } else {
            msg.prevActionConfirmation = "You drew a " + card.getString();
            msg.stackSize = stackSize;

            msg.showBoard = activeBoard;
            msg.dealerHand = dealerHand;
            msg.playerHands = playerHands;
            
            msg.prompt = true;
            msg.actionPrompt = "Option: hit or stand";

            return msg;
        }

    } else if (input == "s") {

        if (!activeBoard) {
            goto invalidLabel;
        }
        
        // Unobscure second card and deal out dealer
        dealerHand.cards[1].setObscured(false);
        while (!dealerHand.busted && dealerHand.values.back() < 17) {
            dealerHand.addCard(shoe.draw());
        }

        // Dealer loses
        if (dealerHand.busted || 
                (dealerHand.values.back() < playerHands[0].values.back()) ) {

            activeBoard = false;

            msg.prevActionConfirmation = "You win!";
            stackSize += 2 * playerHands[0].betAmt;
            msg.stackSize = stackSize;

            msg.showBoard = true;
            msg.dealerHand = dealerHand;
            msg.playerHands = playerHands;
            resetBoard();

            msg.prompt = true;
            msg.actionPrompt = "Option: bet";

            return msg;

        // Draw
        } else if (dealerHand.values.back() == playerHands[0].values.back()) {
            activeBoard = false;

            msg.prevActionConfirmation = "Draw!";
            stackSize += playerHands[0].betAmt;
            msg.stackSize = stackSize;

            msg.showBoard = true;
            msg.dealerHand = dealerHand;
            msg.playerHands = playerHands;
            resetBoard();

            msg.prompt = true;
            msg.actionPrompt = "Option: bet";
            
            return msg;

        // Dealer wins
        } else {
            activeBoard = false;

            msg.prevActionConfirmation = "You lose!";
            msg.stackSize = stackSize;

            msg.showBoard = true;
            msg.dealerHand = dealerHand;
            msg.playerHands = playerHands;
            resetBoard();

            msg.prompt = true;
            msg.actionPrompt = "Option: bet";

            return msg;
        }

    } else if (input == "e") {
        msg.prevActionConfirmation = 
            "Thank you for playing.\nYou bought in for $" + std::to_string(buyIn) + 
            " and ended up with up with $" + std::to_string(stackSize) + 
            ".\nTotal PNL: " + std::to_string((int)stackSize - (int)buyIn);
        msg.stackSize = stackSize;
        msg.showBoard = false;
        msg.prompt = false;
        msg.gameOver = true;

        return msg;

    } else {
        goto invalidLabel;
    }

invalidLabel:
    msg.prevActionConfirmation = "Invalid Action -> Please Try Again";
    msg.stackSize = stackSize;
    msg.showBoard = false;
    if (activeBoard) {
        msg.showBoard = true;
        msg.dealerHand = dealerHand;
        msg.playerHands = playerHands;
    }

    msg.prompt = false;
    // TODO: Add Detailed Response Here 
    return msg;
}
