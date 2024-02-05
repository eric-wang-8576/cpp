#include "game.hpp"
#include <sys/ioctl.h>

std::regex addPattern("a\\s\\$\\d+");
std::regex betPattern("b\\s\\$\\d+");
std::regex betPattern2("b");
std::regex tipPattern("t\\s\\$\\d+");

void Game::resetBoard() {
    Hand hand;
    dealerHand = hand;

    playerHands.clear();
}

void Game::fillMsg(Msg& msg) {
    msg.stackSize = stackSize;
    msg.showBoard = activeBoard;
    msg.dealerHand = dealerHand;
    msg.playerHands = playerHands;
    msg.playerIdx = playerIdx;
    msg.gameOver = gameOver;
};

/*
 * Called when the player adds chips
 */
Msg Game::handleAdd(uint32_t addValue) {
    Msg msg;

    buyIn += addValue;
    stackSize += addValue;

    fillMsg(msg);

    // Populate msg 
    msg.prevActionConfirmation = "You have added $" + 
                                 std::to_string(addValue) + 
                                 " to your stack. Your new stack size is $" +
                                 std::to_string(stackSize) + 
                                 ".";
    msg.prompt = false;
    return msg;
}

/*
 * Called when the player bets or initiates a hand
 * This function only handles the case when it is valid to do so
 *
 */
Msg Game::handleBet(uint32_t betAmt) {
    Msg msg;
    fillMsg(msg);

    std::cout << std::endl;
    for(int i = 0; i < 80; ++i) {
        std::cout << "*";
    }
    std::cout << std::endl;

    // Early return if insufficient chips 
    if (betAmt > stackSize) {
        msg.prevActionConfirmation = "Your current stack size is $" +
                                     std::to_string(stackSize) + 
                                     ". Please enter a smaller bet size or \
                                     add more money.";
        msg.prompt = false;
        return msg;
    }


    // Start the hand 
    activeBoard = true;
    stackSize -= betAmt;

    // Give the dealer a hand, obscuring the second card
    dealerHand.addCard(shoe.draw());
    dealerHand.addCard(shoe.draw());
    dealerHand.obscured = true;

    // Give the player a hand 
    Hand newHand;
    newHand.addCard(shoe.draw());
    newHand.addCard(shoe.draw());
    newHand.betAmt = betAmt;
    playerHands.push_back(newHand);
    playerIdx = 0;

    // Check for blackjack 
    if (dealerHand.isBlackJack || playerHands[0].isBlackJack) {
        playerIdx++;
        return concludeHand();
    }

    // Populate msg 
    fillMsg(msg);
    msg.prevActionConfirmation = "You have bet $" + std::to_string(betAmt) +
                                 ". The board is displayed below.";
    
    msg.prompt = true;
    msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);
    return msg;
}

Msg Game::handleHit() {
    Msg msg;

    Card card = shoe.draw();
    playerHands[playerIdx].addCard(card);

    // If we bust, then let the player know and move onto the next hand 
    if (playerHands[playerIdx].isBusted()) {
        fillMsg(msg);

        playerIdx++;
        // We busted the last hand, so conclude it 
        if (playerIdx == playerHands.size()) {
            return concludeHand();
        }

        msg.prevActionConfirmation = "You bust!";

        msg.prompt = true;
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

        return msg;

    // If no bust, then continue prompting the player
    } else {
        fillMsg(msg);
        msg.prevActionConfirmation = "You drew a " + card.getString();
        
        msg.prompt = true;
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

        return msg;
    }
}

Msg Game::handleStand() {
    Msg msg;

    msg.prevActionConfirmation = "You stood on hand #" + std::to_string(playerIdx + 1);

    playerIdx++;
    // If there are more hands, then request action on them
    if (playerIdx != playerHands.size()) {
        fillMsg(msg);
        msg.prompt = true;
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);
        return msg;
    }

    // If this is the final hand, then resolve the dealer and conclude hand 
    dealerHand.obscured = false;
    while (!dealerHand.isBusted() && dealerHand.values.back() < 17) {
        dealerHand.addCard(shoe.draw());
    }

    return concludeHand();
}

// Assumes that this is a valid position to double in 
Msg Game::handleDouble() {
    Msg msg;

    Card card = shoe.draw();
    playerHands[playerIdx].addCard(card);
    stackSize -= playerHands[playerIdx].betAmt;
    playerHands[playerIdx].betAmt *= 2;

    playerIdx++;
    if (playerIdx == playerHands.size()) {
        dealerHand.obscured = false;
        while (!dealerHand.isBusted() && dealerHand.values.back() < 17) {
            dealerHand.addCard(shoe.draw());
        }

        return concludeHand();

    } else {
        fillMsg(msg);
        msg.prevActionConfirmation = "You doubled down on Hand #" + 
                                     std::to_string(playerIdx);
        msg.prompt = true; 
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

        return msg;
    }
}

// Assumes that this is a valid position to split 
Msg Game::handleSplit() {
    Msg msg;
    
    Hand newHand;
    newHand.addCard(playerHands[playerIdx].cards[1]);
    newHand.betAmt = playerHands[playerIdx].betAmt;
    stackSize -= newHand.betAmt;
    playerHands[playerIdx].popCard();
    playerHands.push_back(newHand);

    fillMsg(msg);
    msg.prevActionConfirmation = "You split Hand #" + std::to_string(playerIdx + 1);
    msg.prompt = true;
    msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

    return msg;
}

Msg Game::concludeHand() {
    Msg msg;

    dealerHand.obscured = false;

    // Calculate payouts 
    uint32_t winnings = 0;
    for (uint8_t idx = 0; idx < playerIdx; ++idx) {
        if (playerHands[idx].isBlackJack && !dealerHand.isBlackJack) {
            winnings += 2 * playerHands[idx].betAmt + playerHands[idx].betAmt / 2;
        } else if (dealerHand.isBusted() ||
                (dealerHand.values.back() < playerHands[idx].values.back())) {
            winnings += 2 * playerHands[idx].betAmt;

        } else if (dealerHand.values.back() == playerHands[idx].values.back()) {
            winnings += playerHands[idx].betAmt;

        }
    }
    stackSize += winnings;

    fillMsg(msg);

    activeBoard = false;
    resetBoard();

    msg.prevActionConfirmation = "Hand over! You receive $" + std::to_string(winnings);
    msg.prompt = true;
    msg.actionPrompt = "Option: bet";

    return msg;
}

Msg Game::processInput(std::string input) {

    // Adding Chips
    if (std::regex_match(input, addPattern)) {
        uint32_t addValue = std::stoi(input.substr(3));

        return handleAdd(addValue);

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

        return handleBet(betAmt);

    // Player hits
    } else if (input == "h") {
        
        if (!activeBoard) {
            goto invalidLabel;
        }

        return handleHit();

    // Player stands
    } else if (input == "s") {

        if (!activeBoard) {
            goto invalidLabel;
        }

        return handleStand();

    // Player doubles down
    } else if (input == "d") {

        if (!activeBoard) {
            goto invalidLabel;
        }

        uint8_t val = playerHands[playerIdx].values.back();
        if (val < 9 || val > 11) {
            goto invalidLabel;
        }

        if (stackSize < playerHands[playerIdx].betAmt) {
            Msg msg;
            fillMsg(msg);
            msg.prevActionConfirmation = "Your current stack size is $" +
                                         std::to_string(stackSize) + 
                                         ". Please add more money \
                                         to perform this action.";
            msg.prompt = false;
            return msg;
        }

        return handleDouble();

    // Player splits 
    } else if (input == "p") {

        if (!activeBoard) {
            goto invalidLabel;
        }

        if (!(playerHands[playerIdx].isPair())) {
            goto invalidLabel;
        }

        if (stackSize < playerHands[playerIdx].betAmt) {
            Msg msg;
            fillMsg(msg);
            msg.prevActionConfirmation = "Your current stack size is $" +
                                         std::to_string(stackSize) + 
                                         ". Please add more money \
                                         to perform this action.";
            msg.prompt = false;
            return msg;
        }

        return handleSplit();

    // Tip the dealer
    } else if (std::regex_match(input, tipPattern)) {
        uint32_t tipValue = std::stoi(input.substr(3));

        if (stackSize < tipValue) {
            Msg msg;
            fillMsg(msg);
            msg.prevActionConfirmation = "Your current stack size is $" +
                                         std::to_string(stackSize) + 
                                         ". Please add more money \
                                         to perform this action.";
            msg.prompt = false;
            return msg;
        }

        stackSize -= tipValue;
        tips += tipValue;

        Msg msg;
        fillMsg(msg);
        msg.prevActionConfirmation = "You tipped the dealer $" + std::to_string(tipValue);
        msg.showBoard = false;
        msg.prompt = false;
        return msg;

    // Player exits
    } else if (input == "e") {
        Msg msg;
        msg.prevActionConfirmation = 
            "Thank you for playing.\nYou bought in for $" + std::to_string(buyIn) + 
            " and ended up with up with $" + std::to_string(stackSize) + 
            ".\nYou tipped $" + std::to_string(tips) +
            ".\n\nTotal PNL: " + std::to_string((int)stackSize - (int)buyIn);
        msg.stackSize = stackSize;
        msg.showBoard = false;
        msg.prompt = false;
        msg.gameOver = true;

        return msg;

    } else {
        goto invalidLabel;
    }

invalidLabel:
    Msg msg;
    fillMsg(msg);
    msg.prevActionConfirmation = "Invalid Action -> Please Try Again";
    msg.prompt = false;
    // TODO: Add Detailed Response Here 
    return msg;
}
