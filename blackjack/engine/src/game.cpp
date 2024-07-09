#include "game.hpp"
#include <sys/ioctl.h>

bool Game::parseAdd(const std::string& input) {
    if (input[0] != 'a' || input[1] != ' ' || input[2] != '$') {
        return false;
    }
    for (int i = 3; i < input.length(); ++i) {
        if (!('0' <= input[i] && input[i] <= '9')) {
            return false;
        }
    }
    return true;
}

// Directly populates the bet states of game if successful
bool Game::parseBet(const std::string& input) {
    uint32_t numBetsP = 0;
    uint32_t totalBetP = 0;
    uint32_t betsP[7];

    auto isDigit = [] (char c) {
        return '0' <= c && c <= '9';
    };

    if (input.length() == 0 || input[0] != 'b') {
        return false;
    }

    uint32_t idx = 1;
    while (idx < input.length() && numBetsP < 7) {
        if (input[idx++] != ' ') {
            return false;
        }
        if (input[idx++] != '$') {
            return false;
        }

        uint32_t currBet = 0;
        while (idx < input.length() && isDigit(input[idx])) {
            currBet *= 10;
            currBet += input[idx] - '0';
            idx++;
        }
        if (currBet == 0) {
            return false;
        }
        betsP[numBetsP] = currBet;
        numBetsP++;
        totalBetP += currBet;
    }

    // There must be at least one bet and at most seven 
    if (numBetsP == 0 || idx != input.length()) {
        return false;
    }

    // Successful bet parse, populate values
    numBets = numBetsP;
    totalBet = totalBetP;
    for (int i = 0; i < numBetsP; ++i) {
        bets[i] = betsP[i];
    }
    return true;
}

bool Game::parseTip(const std::string& input) {
    if (input[0] != 't' || input[1] != ' ' || input[2] != '$') {
        return false;
    }
    for (int i = 3; i < input.length(); ++i) {
        if (!('0' <= input[i] && input[i] <= '9')) {
            return false;
        }
    }
    return true;
}

void Game::resetBoard() {
    // Clear hands
    dealerHand.reset();
    for (uint8_t idx = 0; idx < numPlayerHands; ++idx) {
        playerHands[idx].reset();
    }
    playerIdx = 0;
    numPlayerHands = 0;
}

// Does not populate prevActionConfirmation or actionPrompt fields
void Game::fillMsg(Msg& msg) {
    // Game state
    msg.dealerHandP = &dealerHand;
    msg.playerHandsP = playerHands;
    msg.playerIdx = playerIdx;
    msg.numPlayerHands = numPlayerHands;
    msg.shuffled = false;

    // Display state
    msg.stackSize = stackSize;
    msg.showBoard = activeBoard;
    msg.gameOver = gameOver;
    msg.betInit = false;
};

/*
 * Called when the player adds chips
 */
void Game::handleAdd(uint32_t addValue, Msg& msg) {
    buyIn += addValue;
    stackSize += addValue;

    fillMsg(msg);

    // Populate msg 
    msg.prevActionConfirmation = "You have added " + 
                                 priceToString(addValue) + 
                                 " to your stack. Your new stack size is " +
                                 priceToString(stackSize) + 
                                 ".";
    msg.prompt = false;
    return;
}

/*
 * Called when the player bets or initiates a hand
 * This function only handles the case when it is valid to do so
 *
 */
void Game::handleBet(Msg& msg) {
    resetBoard();

    bool shuffledBeforeHand = shoe.triggerShuffle();

    // Start the hand 
    activeBoard = true;
    numHands++;

    // Deal the dealer two cards, obscuring the second card
    dealerHand.addCard(shoe.draw());
    dealerHand.addCard(shoe.draw());
    dealerHand.setObscured(true);
    dealerHand.setOriginal(true);

    for (uint8_t i = 0; i < numBets; ++i) {
        // Deal the player two cards
        playerHands[numPlayerHands].addCard(shoe.draw());
        playerHands[numPlayerHands].addCard(shoe.draw());
        stackSize -= bets[i];
        playerHands[numPlayerHands].setBetAmt(bets[i]);
        playerHands[numPlayerHands].setOriginal(true);
        numPlayerHands++;
    }

    // Check for blackjack and return early
    if (dealerHand.isBlackjack()) {
        playerIdx = numPlayerHands;
        concludeHand(msg, shuffledBeforeHand);
        return;
    }

    advancePlayerIdx();

    if (playerIdx == numPlayerHands) {
        concludeHand(msg, shuffledBeforeHand);
        return;
    }

    // Populate msg 
    fillMsg(msg);
    msg.prevActionConfirmation = "You bet " + priceToString(totalBet) +
                                 ". The board is displayed below.";
    
    msg.betInit = true;
    msg.prompt = true;
    msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

    if (shuffledBeforeHand) {
        numShuffles++;
        msg.shuffled = true;
        msg.prevActionConfirmation = "The deck was shuffled! " + msg.prevActionConfirmation;
    }

    return;
}

void Game::handleHit(Msg& msg) {
    Card* cardP = shoe.draw();
    playerHands[playerIdx].addCard(cardP);

    // If we bust, then let the player know and move onto the next hand 
    if (playerHands[playerIdx].isBusted()) {

        playerIdx++;
        advancePlayerIdx();
        // We busted the last hand, so conclude it 
        if (playerIdx == numPlayerHands) {
            concludeHand(msg);
            return;
        }

        fillMsg(msg);

        msg.prevActionConfirmation = "You bust!";

        msg.prompt = true;
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

        return;

    // If no bust, then continue prompting the player
    } else {
        fillMsg(msg);
        msg.prevActionConfirmation = "You drew a " + cardP->getString();
        
        msg.prompt = true;
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

        return;
    }
}

void Game::handleStand(Msg& msg) {
    msg.prevActionConfirmation = "You stood on hand #" + std::to_string(playerIdx + 1);

    playerIdx++;
    advancePlayerIdx();
    // If there are more hands, then request action on them
    if (playerIdx != numPlayerHands) {
        fillMsg(msg);
        msg.prompt = true;
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);
        return;
    } else {
        concludeHand(msg);
        return;
    }
}

// Assumes that this is a valid position to double in 
void Game::handleDouble(Msg& msg) {
    // Add a card and double the user's bet
    playerHands[playerIdx].addCard(shoe.draw());
    playerHands[playerIdx].setOriginal(false);
    uint32_t originalBet = playerHands[playerIdx].getBetAmt();
    stackSize -= originalBet;
    playerHands[playerIdx].setBetAmt(originalBet * 2);

    // If we bust, then let the player know and move onto the next hand 
    if (playerHands[playerIdx].isBusted()) {

        playerIdx++;
        advancePlayerIdx();
        // We busted the last hand, so conclude it 
        if (playerIdx == numPlayerHands) {
            concludeHand(msg);
            return;
        }

        fillMsg(msg);

        msg.prevActionConfirmation = "You bust!";

        msg.prompt = true;
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

        return;
    }

    playerIdx++;
    advancePlayerIdx();
    if (playerIdx == numPlayerHands) {
        concludeHand(msg);
        return;

    } else {
        fillMsg(msg);
        msg.prevActionConfirmation = "You doubled down on Hand #" + 
                                     std::to_string(playerIdx);
        msg.prompt = true; 
        msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

        return;
    }
}

// Assumes that this is a valid position to split 
void Game::handleSplit(Msg& msg) {
    bool splitAces = playerHands[playerIdx].areAces();

    // The new hand points to the second card
    Card* cardP = playerHands[playerIdx].popCard();
    playerHands[playerIdx].setOriginal(false);

    // Shift all remaining cards upward
    for (int idx = numPlayerHands; idx > playerIdx + 1; --idx) {
        playerHands[idx] = playerHands[idx - 1];
    }

    playerHands[playerIdx + 1].reset();
    playerHands[playerIdx + 1].addCard(cardP);
    playerHands[playerIdx + 1].setOriginal(false);

    // Double the user's bet
    uint32_t originalBet = playerHands[playerIdx].getBetAmt();
    stackSize -= originalBet;
    playerHands[playerIdx + 1].setBetAmt(originalBet);

    // Increment numPlayerHands to match number of hands
    numPlayerHands++;

    // If aces were split, deal both cards
    if (splitAces) {
        playerHands[playerIdx].addCard(shoe.draw());
        playerHands[playerIdx + 1].addCard(shoe.draw());
        playerIdx += 2;
    }

    if (playerIdx == numPlayerHands) {
        concludeHand(msg);
        return;
    }

    fillMsg(msg);
    msg.prevActionConfirmation = "You split Hand #" + std::to_string(playerIdx + 1);
    msg.prompt = true;
    msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

    return;
}

void Game::handleTip(uint32_t tipValue, Msg& msg) {
    stackSize -= tipValue;
    tips += tipValue;

    fillMsg(msg);
    msg.prevActionConfirmation = "You tipped the dealer " + priceToString(tipValue);
    msg.showBoard = false;
    msg.prompt = false;

    return;
}

void Game::handleExit(Msg& msg) {
    int pnl = (int) stackSize - (int) buyIn;
    msg.prevActionConfirmation = 
        "Thank you for playing.\nYou bought in for " + priceToString(buyIn) + 
        " and ended up with up with " + priceToString(stackSize) + 
        ".\nYou tipped " + priceToString(tips) +
        ".\nYou played a total of " + valueToString(numHands) + " hands" +
        " and the deck was shuffled " + valueToString(numShuffles) + " times." +
        "\n\nTotal PNL: " + PNLToString(pnl);
    msg.stackSize = stackSize;
    msg.showBoard = false;
    msg.prompt = false;
    msg.gameOver = true;
    msg.PNL = (int) stackSize - (int) buyIn;

    return;
}

void Game::handleInvalid(Msg& msg) {
    fillMsg(msg);
    msg.prevActionConfirmation = "Invalid Action -> Please Try Again";
    msg.prompt = false;

    return;
}

void Game::handleInsufficientFunds(Msg& msg) {
    fillMsg(msg);
    msg.prevActionConfirmation = "Your current stack size is " +
                                 priceToString(stackSize) + 
                                 ". Please add more money " +
                                 "to perform this action.";
    msg.prompt = false;

    return;
}

void Game::advancePlayerIdx() {
    while (playerIdx < numPlayerHands && playerHands[playerIdx].isBlackjack()) {
        playerIdx++;
    }
}

void Game::concludeHand(Msg& msg, bool justShuffled) {
    // We deal the dealer's cards only if the player has at least one hand
    //   that is not blackjack or busted
    bool dealDealerCards = false;
    for (uint8_t idx = 0; idx < playerIdx; ++idx) {
        if (!(playerHands[idx].isBusted() || playerHands[idx].isBusted())) {
            dealDealerCards = true;
            break;
        }
    }

    // Deal the dealer's hands if the player has at least one non-busted hand and it is not a blackjack 
    if (dealDealerCards) {
        while (dealerHand.shouldDealerHit()) {
            dealerHand.addCard(shoe.draw());
            dealerHand.setOriginal(false);
        }
    }

    dealerHand.setObscured(false);

    // Calculate payouts 
    uint32_t winnings = 0; // How much the player ends up winning, 
                           // including what they initially invested
    uint32_t invested = 0; // How much the player invested into the hand

    for (uint8_t idx = 0; idx < playerIdx; ++idx) {
        if (playerHands[idx].isBlackjack() && !dealerHand.isBlackjack()) {
            // Player blackjack
            winnings += 2 * playerHands[idx].getBetAmt() + playerHands[idx].getBetAmt() / 2;

        } else if (playerHands[idx].isBusted() || dealerHand.getPrimaryVal() > playerHands[idx].getPrimaryVal()) {
            // The player loses entirely

        } else if (dealerHand.isBusted() || (dealerHand.getPrimaryVal() < playerHands[idx].getPrimaryVal())) {
            // The player wins entirely 
            winnings += 2 * playerHands[idx].getBetAmt();

        } else if (dealerHand.getPrimaryVal() == playerHands[idx].getPrimaryVal()) {
            // The player pushes
            winnings += playerHands[idx].getBetAmt();

        } else {
            std::cout << "Payout Error" << std::endl;
            exit(1);
        }

        invested += playerHands[idx].getBetAmt();
        playerHands[idx].setBetAmt(0);
    }
    stackSize += winnings;


    // Populate message
    fillMsg(msg);

    activeBoard = false;

    if (winnings > invested) {
        msg.prevActionConfirmation = "You win! You receive " + priceToString(winnings);
    } else if (winnings == invested) {
        msg.prevActionConfirmation = "Draw! You receive " + priceToString(winnings);
    } else {
        msg.prevActionConfirmation = "Dealer wins! You receive " + priceToString(winnings);
    }

    msg.prompt = true;
    msg.actionPrompt = "Option: bet";

    // This should only be true when we bet and either player or dealer has blackjack 
    if (justShuffled) {
        numShuffles++;
        msg.shuffled = true;
        msg.prevActionConfirmation = "The deck was shuffled! " + msg.prevActionConfirmation;
    }

    return;
}

void Game::processInput(const std::string& input, Msg& msg) {

    // Bet
    if (input == "b" || parseBet(input)) {
        if (activeBoard) {
            return handleInvalid(msg);
        }

        if (totalBet > stackSize) {
            return handleInsufficientFunds(msg);
        }

        return handleBet(msg);

    // Hit
    } else if (input == "h") {
        
        if (!activeBoard) {
            return handleInvalid(msg);
        }

        return handleHit(msg);

    // Stand
    } else if (input == "s") {

        if (!activeBoard) {
            return handleInvalid(msg);
        }

        return handleStand(msg);

    // Double down
    } else if (input == "d") {

        if (!activeBoard) {
            return handleInvalid(msg);
        }

        if (playerHands[playerIdx].getNumCards() != 2) {
            return handleInvalid(msg);
        }

        if (stackSize < playerHands[playerIdx].getBetAmt()) {
            return handleInsufficientFunds(msg);
        }

        return handleDouble(msg);

    // Split
    } else if (input == "p") {

        if (!activeBoard) {
            return handleInvalid(msg);
        }

        if (!(playerHands[playerIdx].isPair())) {
            return handleInvalid(msg);
        }

        if (stackSize < playerHands[playerIdx].getBetAmt()) {
            return handleInsufficientFunds(msg);
        }

        return handleSplit(msg);

    // Add more money
    } else if (parseAdd(input)) {
        uint32_t addValue;
        try {
            addValue = std::stoi(input.substr(3));
        } catch (const std::out_of_range& e) {
            return handleInvalid(msg);
        }

        return handleAdd(addValue, msg);

    // Tip
    } else if (parseTip(input)) {
        uint32_t tipValue;
        try {
            tipValue = std::stoi(input.substr(3));
        } catch (const std::out_of_range& e) {
            return handleInvalid(msg);
        }

        if (stackSize < tipValue) {
            return handleInsufficientFunds(msg);
        }

        return handleTip(tipValue, msg);

    // Player exits
    } else if (input == "e") {

        return handleExit(msg);

    } else {
        return handleInvalid(msg);
    }
}
