#include "game.hpp"
#include <sys/ioctl.h>

std::regex addPattern("a\\s\\$\\d+");
std::regex betPattern("b\\s\\$\\d+");
std::regex betPattern2("b");
std::regex tipPattern("t\\s\\$\\d+");

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
    msg.prevActionConfirmation = "You have added $" + 
                                 std::to_string(addValue) + 
                                 " to your stack. Your new stack size is $" +
                                 std::to_string(stackSize) + 
                                 ".";
    msg.prompt = false;
    return;
}

/*
 * Called when the player bets or initiates a hand
 * This function only handles the case when it is valid to do so
 *
 */
void Game::handleBet(uint32_t betAmt, Msg& msg) {
    resetBoard();
    
    // Early return if insufficient chips 
    if (betAmt > stackSize) {
        msg.prevActionConfirmation = "Your current stack size is $" +
                                     std::to_string(stackSize) + 
                                     ". Please enter a smaller bet size or " +
                                     "add more money.";
        msg.prompt = false;
        msg.showBoard = false;
        return;
    }

    bool shuffledBeforeHand = shoe.triggerShuffle();

    // Start the hand 
    activeBoard = true;
    stackSize -= betAmt;
    numHands++;

    // Deal the dealer two cards, obscuring the second card
    dealerHand.addCard(shoe.draw());
    dealerHand.addCard(shoe.draw());
    dealerHand.setObscured(true);

    // Deal the player two cards
    playerHands[numPlayerHands].addCard(shoe.draw());
    playerHands[numPlayerHands].addCard(shoe.draw());
    playerHands[numPlayerHands].setBetAmt(betAmt);
    numPlayerHands++;

    // Check for blackjack and return early
    if (dealerHand.isBlackjack() || playerHands[0].isBlackjack()) {
        playerIdx++;
        concludeHand(msg, shuffledBeforeHand);
        return;
    }

    // Populate msg 
    fillMsg(msg);
    msg.prevActionConfirmation = "You bet $" + std::to_string(betAmt) +
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
    uint32_t originalBet = playerHands[playerIdx].getBetAmt();
    stackSize -= originalBet;
    playerHands[playerIdx].setBetAmt(originalBet * 2);


    // If we bust, then let the player know and move onto the next hand 
    if (playerHands[playerIdx].isBusted()) {

        playerIdx++;
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
    playerHands[numPlayerHands].addCard(cardP);

    // Double the user's bet
    uint32_t originalBet = playerHands[playerIdx].getBetAmt();
    stackSize -= originalBet;
    playerHands[numPlayerHands].setBetAmt(originalBet);

    // Increment numPlayerHands to match number of hands
    numPlayerHands++;

    // If aces were split, deal both cards and resolve action 
    if (splitAces) {
        playerHands[playerIdx].addCard(shoe.draw());
        playerHands[playerIdx + 1].addCard(shoe.draw());
        playerIdx += 2;

        concludeHand(msg);
        return;
    }

    fillMsg(msg);
    msg.prevActionConfirmation = "You split Hand #" + std::to_string(playerIdx + 1);
    msg.prompt = true;
    msg.actionPrompt = "Option: action on hand #" + std::to_string(playerIdx + 1);

    return;
}

void Game::concludeHand(Msg& msg, bool justShuffled) {
    bool oneNoneBusted = false;
    for (uint8_t idx = 0; idx < playerIdx; ++idx) {
        if (!playerHands[idx].isBusted()) {
            oneNoneBusted = true;
            break;
        }
    }

    // Deal the dealer's hands if the player has at least one non-busted hand and it is not a blackjack 
    if (oneNoneBusted && !(playerIdx == 1 && playerHands[0].isBlackjack())) {
        while (dealerHand.shouldDealerHit()) {
            dealerHand.addCard(shoe.draw());
        }
    }

    dealerHand.setObscured(false);

    // Calculate payouts 
    uint32_t winnings = 0; // How much the player ends up winning, 
                           // including what they initially invested
    uint32_t invested = 0; // How much the player invested into the hand

    for (uint8_t idx = 0; idx < playerIdx; ++idx) {
        if (playerIdx == 1 && playerHands[idx].isBlackjack() && !dealerHand.isBlackjack()) {
            // Player blackjack, only pay out if obtained from the first two cards
            winnings += 2 * playerHands[idx].getBetAmt() + playerHands[idx].getBetAmt() / 2;

        } else if (playerHands[idx].isBusted() || dealerHand.getPrimaryVal() > playerHands[idx].getPrimaryVal()) {
            // The player loses entirely

        } else if (dealerHand.isBusted() ||
                (dealerHand.getPrimaryVal() < playerHands[idx].getPrimaryVal())) {
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
        msg.prevActionConfirmation = "You win! You receive $" + std::to_string(winnings);
    } else if (winnings == invested) {
        msg.prevActionConfirmation = "Draw! You receive $" + std::to_string(winnings);
    } else {
        msg.prevActionConfirmation = "Dealer wins! You receive $" + std::to_string(winnings);
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

void Game::processInput(std::string input, Msg& msg) {

    // Adding Chips
    if (std::regex_match(input, addPattern)) {
        uint32_t addValue = std::stoi(input.substr(3));

        return handleAdd(addValue, msg);

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

        return handleBet(betAmt, msg);

    // Player hits
    } else if (input == "h") {
        
        if (!activeBoard) {
            goto invalidLabel;
        }

        return handleHit(msg);

    // Player stands
    } else if (input == "s") {

        if (!activeBoard) {
            goto invalidLabel;
        }

        return handleStand(msg);

    // Player doubles down
    } else if (input == "d") {

        if (!activeBoard) {
            goto invalidLabel;
        }

        if (playerHands[playerIdx].getNumCards() != 2) {
            goto invalidLabel;
        }

        if (stackSize < playerHands[playerIdx].getBetAmt()) {
            fillMsg(msg);
            msg.prevActionConfirmation = "Your current stack size is $" +
                                         std::to_string(stackSize) + 
                                         ". Please add more money " +
                                         "to perform this action.";
            msg.prompt = false;
            return;
        }

        return handleDouble(msg);

    // Player splits 
    } else if (input == "p") {

        if (!activeBoard) {
            goto invalidLabel;
        }

        if (!(playerHands[playerIdx].isPair())) {
            goto invalidLabel;
        }

        if (stackSize < playerHands[playerIdx].getBetAmt()) {
            fillMsg(msg);
            msg.prevActionConfirmation = "Your current stack size is $" +
                                         std::to_string(stackSize) + 
                                         ". Please add more money " +
                                         "to perform this action.";
            msg.prompt = false;
            return;
        }

        return handleSplit(msg);

    // Tip the dealer
    } else if (std::regex_match(input, tipPattern)) {
        uint32_t tipValue = std::stoi(input.substr(3));

        if (stackSize < tipValue) {
            fillMsg(msg);
            msg.prevActionConfirmation = "Your current stack size is $" +
                                         std::to_string(stackSize) + 
                                         ". Please add more money " +
                                         "to perform this action.";
            msg.prompt = false;
            return;
        }

        stackSize -= tipValue;
        tips += tipValue;

        fillMsg(msg);
        msg.prevActionConfirmation = "You tipped the dealer $" + std::to_string(tipValue);
        msg.showBoard = false;
        msg.prompt = false;
        return;

    // Player exits
    } else if (input == "e") {
        int pnl = (int) stackSize - (int) buyIn;
        msg.prevActionConfirmation = 
            "Thank you for playing.\nYou bought in for $" + std::to_string(buyIn) + 
            " and ended up with up with $" + std::to_string(stackSize) + 
            ".\nYou tipped $" + std::to_string(tips) +
            ".\nYou played a total of " + std::to_string(numHands) + " hands" +
            " and the deck was shuffled " + std::to_string(numShuffles) + " times." +
            "\n\nTotal PNL: " + (pnl < 0 ? "-" : "+") + "$" + std::to_string(pnl > 0 ? pnl : -pnl);
        msg.stackSize = stackSize;
        msg.showBoard = false;
        msg.prompt = false;
        msg.gameOver = true;
        msg.PNL = (int) stackSize - (int) buyIn;

        return;

    } else {
        goto invalidLabel;
    }

invalidLabel:
    fillMsg(msg);
    msg.prevActionConfirmation = "Invalid Action -> Please Try Again";
    msg.prompt = false;
    return;
}
