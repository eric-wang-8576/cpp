// reading a text file
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <limits.h>
using namespace std;

#define MAX_HOLE_CARDS 4
#define NUM_BOARD_CARDS 5
#define MAX_DOUBLE

typedef string Card;

vector<string> split(const string &s, char delim) {
  vector<string> result;
  stringstream ss (s);
  string item;

  while (getline(ss, item, delim)) {
    result.push_back(item);
  }

  return result;
}

void throwError(const string &s) {
  cout << s << endl;
  exit(1);
}

int getCardRank(string s) {
  if (s == "2") { return 0; }
  else if (s == "3") { return 1; }
  else if (s == "4") { return 2; }
  else if (s == "5") { return 3; }
  else if (s == "6") { return 4; }
  else if (s == "7") { return 5; }
  else if (s == "8") { return 6; }
  else if (s == "9") { return 7; }
  else if (s == "10") { return 8; }
  else if (s == "j" || s == "J") { return 9; }
  else if (s == "q" || s == "Q") { return 10; }
  else if (s == "k" || s == "K") { return 11; }
  else if (s == "a" || s == "A") { return 12; }
  else { cout << "INVALID CARD!" << s; return 0; }
}

int getSuitRank(string s) {
  if (s == "c" || s == "C") { return 0; }
  else if (s == "d" || s == "D") { return 1; }
  else if (s == "h" || s == "H") { return 2; }
  else if (s == "s" || s == "S") { return 3; }
  else { cout << "INVALID CARD!" << s; return 0; }
}

bool isQuads(int* valFreq) {
  for (int i = 0; i < 13; i++) {
    if (valFreq[i] == 4) {
      return true;
    }
  }
  return false;
}

bool isFullHouse(int* valFreq) {
  bool hasTwo = false;
  bool hasThree = false;
  for (int i = 0; i < 13; i++) {
    if (valFreq[i] == 2){
      hasTwo = true;
    }
    if (valFreq[i] == 3) {
      hasThree = true;
    }
  }
  return hasTwo && hasThree;
}

bool isStraight(int* valFreq) {
  // Wheel
  if (valFreq[12] == 1) {
    bool isStraight = true;
    for (int i = 0; i < 4; i++) {
      if (valFreq[i] != 1) {
        isStraight = false;
        break;
      }
    }
    if (isStraight) {
      return true;
    }
  }
  // Normal Straight
  for (int i = 0; i < 13; i++) {
    if (valFreq[i] == 1) {
      for (int j = i + 1; j < i + 5; j++) {
        if (j > 12 || valFreq[j] != 1) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

bool isFlush(int* suitFreq) {
  for (int i = 0; i < 4; i++) {
    if (suitFreq[i] == 5) {
      return true;
    }
  }
  return false;
}

bool isThree(int* valFreq) {
  bool hasTwo = false;
  bool hasThree = false;
  for (int i = 0; i < 13; i++) {
    if (valFreq[i] == 2){
      hasTwo = true;
    } else if (valFreq[i] == 3) {
      hasThree = true;
    }
  }
  return (!hasTwo) && hasThree;
}

bool isTwoPair(int* valFreq) {
  int hasTwo = 0;
  for (int i = 0; i < 13; i++) {
    if (valFreq[i] == 2){
      hasTwo += 1;
    }
  }
  return hasTwo == 2;
}

bool isPair(int* valFreq) {
  int hasTwo = 0;
  for (int i = 0; i < 13; i++) {
    if (valFreq[i] == 2){
      hasTwo += 1;
    }
  }
  return hasTwo == 1;
}

#define HAND_RANK 10000000000L
#define VALUE1 100000000L
#define VALUE2 1000000L
#define KICKER1 10000L
#define KICKER2 100L
#define KICKER3 1L

long getScoreOfHand(string* combinedCards) {
  int ret;
  int* valFreq = new int[13];
  for (int i = 0; i < 13; i++) {
    valFreq[i] = 0;
  }
  int* suitFreq = new int[4]; 
  for (int i = 0; i < 4; i++) {
    suitFreq[i] = 0;
  }

  for (int i = 0; i < 5; i++) {
    string currCard = combinedCards[i];
    int length = currCard.length();
    valFreq[getCardRank(currCard.substr(0, length - 1))]++;
    suitFreq[getSuitRank(currCard.substr(length - 1, 1))]++;
  }

  // for (int i = 0; i < 13; i++) {
  //   cout << valFreq[i] << " ";
  // }
  // cout << "\n";

  // for (int i = 0; i < 4; i++) {
  //   cout << suitFreq[i] << " ";
  // }
  // cout << "\n";
  
  // WHILE ENCODING, ADD TWO
  if (isFlush(suitFreq) && isStraight(valFreq)) { // Straight Flush
    for (int i = 0; i < 13; i++) {
      if (valFreq[i] == 1) {
        return HAND_RANK*9 + VALUE1*(i + 2);
      }
    }
  } else if (isQuads(valFreq)) { // Quads
    long QUAD_VAL = 0;
    long KICKER_VAL = 0;
    for (int i = 12; i >= 0; i--) {
      if (valFreq[i] == 4) {
        QUAD_VAL = i + 2;
      } else if (valFreq[i] == 1) {
        KICKER_VAL = i + 2;
      }
    }
    return HAND_RANK*8 + VALUE1*QUAD_VAL + KICKER1*KICKER_VAL;
  } else if (isFullHouse(valFreq)) { // Full House
    long THREE_VAL = 0;
    long TWO_VAL = 0;
    for (int i = 12; i >= 0; i--) {
      if (valFreq[i] == 3) {
        THREE_VAL = i + 2;
      } else if (valFreq[i] == 2) {
        TWO_VAL = i + 2;
      }
    }
    return HAND_RANK*7 + VALUE1*THREE_VAL + VALUE2*TWO_VAL;
  } else if (isFlush(suitFreq)) { // Flush
    long FIRST = 0;
    long SECOND = 0;
    long THIRD = 0;
    long FOURTH = 0;
    long FIFTH = 0;
    for (int i = 12; i >= 0; i--) {
      if (valFreq[i] == 1) {
        if (FIRST == 0) {
          FIRST = i + 2;
        } else if (SECOND == 0) {
          SECOND = i + 2;
        } else if (THIRD == 0) {
          THIRD = i + 2;
        } else if (FOURTH == 0) {
          FOURTH = i + 2;
        } else {
          FIFTH = i + 2;
        }
      } 
    }
    return HAND_RANK*6 + VALUE1*FIRST + VALUE2*SECOND + KICKER1*THIRD + KICKER2*FOURTH + KICKER3*FIFTH;
  } else if (isStraight(valFreq)) { // Straight
    for (int i = 0; i < 13; i++) {
      if (valFreq[i] == 1) {
        if (i == 0 && valFreq[12] == 1) {
          return HAND_RANK*5 + VALUE1*(i + 1);
        } else {
          return HAND_RANK*5 + VALUE1*(i + 2);
        }
      }
    }
  } else if (isThree(valFreq)) { // Trips/Set
    long THREE_VAL = 0;
    long KICKER_1 = 0;
    long KICKER_2 = 0;
    for (int i = 12; i >= 0; i--) {
      if (valFreq[i] == 1) {
        if (KICKER_1 == 0) {
          KICKER_1 = i + 2;
        } else if (KICKER_2 == 0) {
          KICKER_2 = i + 2;
        }
      } else if (valFreq[i] == 3) {
        THREE_VAL = i + 2;
      }
    }
    return HAND_RANK*4 + VALUE1*THREE_VAL + KICKER1*KICKER_1 + KICKER2*KICKER_2;
  } else if (isTwoPair(valFreq)) { // Two Pair
    long VAL1 = 0;
    long VAL2 = 0;
    long KICKER = 0;
    for (int i = 12; i >= 0; i--) {
      if (valFreq[i] == 2) {
        if (VAL1 == 0) {
          VAL1 = i + 2;
        } else if (VAL2 == 0) {
          VAL2 = i + 2;
        }
      } else if (valFreq[i] == 1) {
        KICKER = i + 2;
      }
    }
    return HAND_RANK*3 + VALUE1*VAL1 + VALUE2*VAL2 + KICKER1*KICKER;
  } else if (isPair(valFreq)) { // One Pair
    long VAL1 = 0;
    long KICKER_1 = 0;
    long KICKER_2 = 0;
    long KICKER_3 = 0;
    for (int i = 12; i >= 0; i--) {
      if (valFreq[i] == 2) {
        VAL1 = i + 2;
      } else if (valFreq[i] == 1) {
        if (KICKER_1 == 0) {
          KICKER_1 = i + 2;
        } else if (KICKER_2 == 0) {
          KICKER_2 = i + 2;
        } else if (KICKER_3 == 0) {
          KICKER_3 = i + 2;
        }
      }
    }
    return HAND_RANK*2 + VALUE1*VAL1 + KICKER1*KICKER_1 + KICKER2*KICKER_2 + KICKER3*KICKER_3;
  } else { // High Card
    long FIRST = 0;
    long SECOND = 0;
    long THIRD = 0;
    long FOURTH = 0;
    long FIFTH = 0;
    for (int i = 12; i >= 0; i--) {
      if (valFreq[i] == 1) {
        if (FIRST == 0) {
          FIRST = i + 2;
        } else if (SECOND == 0) {
          SECOND = i + 2;
        } else if (THIRD == 0) {
          THIRD = i + 2;
        } else if (FOURTH == 0) {
          FOURTH = i + 2;
        } else {
          FIFTH = i + 2;
        }
      } 
    }
    return HAND_RANK*1 + VALUE1*FIRST + VALUE2*SECOND + KICKER1*THIRD + KICKER2*FOURTH + KICKER3*FIFTH;
  }
  return 0;
}

class HoleCards {
  public:
    Card* cards;
    int numCards;

    HoleCards () { // INITIALIZED FORMALLY IN PLAYER CONSTRUCTOR 
      cards = new Card[MAX_HOLE_CARDS];
    }

    void printHoleCards() {
      cout << "Printing hole cards";
      for (int i = 0; i < numCards; i++) {
        cout << " " << cards[i];
      }
      cout << "\n";
    }
};

class Board {
  public:
    Card* cards;

    Board (string s) {
      vector<Card> vec = split(s, ' ');
      
      if (vec.size() != 5) {
        throwError("Wrong number of cards on the board!");
      }

      cards = new Card[NUM_BOARD_CARDS];
      for (int i = 0; i < NUM_BOARD_CARDS; i++) {
        cards[i] = vec[i];
      }
    }

    void printBoard() {
      cout << "Printing New Board: ";
      for (int i = 0; i < NUM_BOARD_CARDS; i++) {
        cout << " " << cards[i];
      }
      cout << "\n" << "\n";

      // PRINT GENERATED CARDS
      vector<vector<Card>> generatedCards = generatePossibleThreeCards();
      cout << "Generating Possible Three Cards For Board:" << endl;
      for (vector<Card> innerVec : generatedCards) {
        cout << "New Card: ";
        for (Card innerCard: innerVec) {
          cout << " " << innerCard;
        }
        cout << "\n";
      }
      cout << "\n";

      // PRINT GENERATED CARDS
      generatedCards = generatePossibleFourCards();
      cout << "Generating Possible Four Cards For Board:" << endl;
      for (vector<Card> innerVec : generatedCards) {
        cout << "New Card: ";
        for (Card innerCard: innerVec) {
          cout << " " << innerCard;
        }
        cout << "\n";
      }
      cout << "\n";

    }

    vector<vector<Card>> generatePossibleThreeCards() {
      vector<vector<Card>> ans = vector<vector<Card>>();
      for (int i = 0; i < NUM_BOARD_CARDS; i++) {
        for (int j = 0; j < i; j++) {
          for (int k = 0; k < j; k++) {
            vector<Card> newVec = vector<Card>();
            newVec.push_back(cards[i]);
            newVec.push_back(cards[j]);
            newVec.push_back(cards[k]);
            ans.push_back(newVec);
          }
        }
      }
      return ans;
    }

    vector<vector<Card>> generatePossibleFourCards() {
      vector<vector<Card>> ans = vector<vector<Card>>();
      for (int i = 0; i < NUM_BOARD_CARDS; i++) {
        for (int j = 0; j < i; j++) {
          for (int k = 0; k < j; k++) {
            for (int l = 0; l < k; l++) {
              vector<Card> newVec = vector<Card>();
              newVec.push_back(cards[i]);
              newVec.push_back(cards[j]);
              newVec.push_back(cards[k]);
              newVec.push_back(cards[l]);
              ans.push_back(newVec);
            }
          }
        }
      }
      return ans;
    }
};

class Player {
  public:
    string name;
    double originalStackSize;
    double stackSize;
    HoleCards holeCards;
    int highestScore;

    bool processed;
    double payout;

    Player (string s) {
      vector<string> vec = split(s, ' ');
      name = vec[0];
      stackSize = stod(vec[1]); 
      originalStackSize = stackSize;

      holeCards = HoleCards();
      for (int i = 2; i < vec.size(); i++) {
        holeCards.cards[i - 2] = vec[i];
      }
      holeCards.numCards = vec.size() - 2;

      highestScore = 0;
      
      processed = false;
      payout = 0;
    }

    void printPlayer() {
      cout << "Player name is: " << name << endl;
      cout << "Player highest score is: " << highestScore << endl;
      cout << "Player stack size is: " << fixed << setprecision(2) << stackSize << endl;
      holeCards.printHoleCards();
      cout << "\n";

      // PRINT GENERATED CARDS
      vector<vector<Card>> generatedCards = generatePossibleTwoCards();
      cout << "Generating Possible Two Cards For Player:" << endl;
      for (vector<Card> innerVec : generatedCards) {
        cout << "New Card: ";
        for (Card innerCard: innerVec) {
          cout << " " << innerCard;
        }
        cout << "\n";
      }
      cout << "\n";

      generatedCards = generatePossibleOneCard();
      cout << "Generating Possible One Card For Player:" << endl;
      for (vector<Card> innerVec : generatedCards) {
        cout << "New Card: ";
        for (Card innerCard: innerVec) {
          cout << " " << innerCard;
        }
        cout << "\n";
      }
      cout << "\n";
    }

    vector<vector<Card>> generatePossibleTwoCards() {
      vector<vector<Card>> ans = vector<vector<Card>>();
      int numHoleCards = holeCards.numCards; 
      for (int i = 0; i < numHoleCards; i++) {
        for (int j = 0; j < i; j++) {
          vector<Card> newVec = vector<Card>();
          newVec.push_back(holeCards.cards[i]);
          newVec.push_back(holeCards.cards[j]);
          ans.push_back(newVec);
        }
      }
      return ans;
    }

    vector<vector<Card>> generatePossibleOneCard() {
      vector<vector<Card>> ans = vector<vector<Card>>();
      int numHoleCards = holeCards.numCards; 
      for (int i = 0; i < numHoleCards; i++) {
        vector<Card> newVec = vector<Card>();
        newVec.push_back(holeCards.cards[i]);
        ans.push_back(newVec);
      }
      return ans;
    }
};

class Showdown {
  private:
    int numPlayers;
    int numBoards;
    string type;
    string currFile;

  public:
    vector<Player> players;
    vector<Board> boards;

    Showdown() {
      numPlayers = 0;
      numBoards = 0;
      players = vector<Player>();
      boards = vector<Board>();
    }

    void parseInput(string fileName) {
      currFile = fileName;
      string currLine;
      ifstream myfile (fileName);

      // capture type

      getline(myfile, currLine);
      if (currLine != "PLO" && currLine != "NLH") {
        throwError("Invalid type!");
      }
      setType(currLine);

      // capture boards

      getline(myfile, currLine);
      setNumBoards(stoi(currLine));

      for (int i = 0; i < getNumBoards(); i++) {
        getline(myfile, currLine);
        addBoard(currLine);
      }

      printBoards();

      // capture players

      getline(myfile, currLine);
      setNumPlayers(stoi(currLine));

      for (int i = 0; i < getNumPlayers(); i++) {
        getline(myfile, currLine);
        addPlayer(currLine);
      }

      printPlayers();
    }

    int numUnprocessed() {
      int count = 0;
      for (Player player : players) {
        if (player.processed == false) {
          count += 1;
        }
      }
      return count;
    }

    void calculatePayouts() {

      int sidePotNum = 1;
      while (numUnprocessed() > 1) {
        // cout << numUnprocessed();
        // Get smallest value and corresponding player
        double smallestStack = INT_MAX;
        Player* playerToProcess = nullptr;
        for (Player& player: players) {
          if (player.processed == false && player.stackSize < smallestStack) {
            smallestStack = player.stackSize;
            playerToProcess = &player;
          }
        }

        // Get the players involved in this round and get their values
        double currPot = 0;
        vector<Player*> involved = vector<Player*>();
        for (Player& player: players) {
          Player* player_p = &player;
          if (player_p->processed == false) {
            involved.push_back(player_p);
            player_p->stackSize -= smallestStack;
            if (player_p->stackSize == 0) { // In case someone else drops to zero, we can mark them as processed too
              player_p->processed = true;
            }
            currPot += smallestStack;
          }
        }

        // Mark player as processed
        (*playerToProcess).processed = true;

        cout << "SIDE POT #" << sidePotNum++ << ": " << currPot << endl;
        cout << "Participants: ";
        for (Player* player : involved) {
          cout << player->name << " ";
        }
        cout << "\n";

        // For each board, calculate and distribute the payouts 
        currPot /= numBoards;
        int boardNum = 1;
        for (int boardInd = 0; boardInd < numBoards; boardInd++) {
          vector<Player*> winners = vector<Player*>();
          long maxScore = 0;
          for (Player* player_p: involved) {
            long score = calculatePlayerBoardScore(*player_p, boards[boardInd]);
            if (score > maxScore) {
              maxScore = score;
              winners.clear();
              winners.push_back(player_p);
            } else if (score == maxScore) {
              winners.push_back(player_p);
            }
          } 

          int numWinners = winners.size();
          for (Player* player : winners) {
            player->payout += currPot/winners.size();
          }

          cout << "Board " << boardNum++ << " Winners: ";
          for (Player* player : winners) {
            cout << player->name << " ";
          }
          cout << "\n";
        }
        cout << "\n";

        
      }

      for (Player& player: players) {
        double remaining = player.stackSize;
        player.stackSize -= remaining;
        player.payout += remaining;
      }
      
    }

    void printPayouts() {
      ofstream myFile;
      myFile.open(currFile, ios_base::app);
      string toWrite;

      myFile << "\n\n\nFINAL PAYOUTS:" << endl;
      cout << "\n\nFINAL PAYOUTS:" << endl;
      for (Player player: players) {
        myFile << player.name << ": " << player.originalStackSize << " -> " << player.payout << " (" << showpos << player.payout - player.originalStackSize << noshowpos << ")" << endl;
        cout << player.name << ": " << player.originalStackSize << " -> " << player.payout << " (" << showpos << player.payout - player.originalStackSize << noshowpos << ")" << endl;
      }
      myFile << "\n";
      cout << "\n\n\n";
    }

    string getHandFromScore(long score) {
      int handType = score / HAND_RANK;
      switch (handType) {
        case (1):
          return "High Card";
        case (2):
          return "One Pair";
        case (3):
          return "Two Pair";
        case (4):
          return "Trips/Set";
        case (5):
          return "Straight";
        case (6):
          return "Flush";
        case (7):
          return "Full House";
        case (8):
          return "Quads";
        case (9):
          return "Straight Flush";
        default:
          throwError("Hand is not possible!");
      };
      return "";
    }

    void calculateScores() {
      cout << "PLAYER/BOARD HAND - SCORES: " << "\n";
      for (Player player : players) {
        int boardNum = 1;
        for (Board board : boards) {
          long score = calculatePlayerBoardScore(player, board);
          cout << player.name << "/Board " << boardNum++ << ": " << getHandFromScore(score) << " - " << score << endl;
        }
        cout << "\n\n";
      }
    }

    long calculatePlayerBoardScore(Player player, Board board) {
      long score = 0;
      string* combined = new string[5];
      // Both NLH and PLO have this logic 
      vector<vector<Card>> playerPossibleCards = player.generatePossibleTwoCards();
      vector<vector<Card>> boardPossibleCards = board.generatePossibleThreeCards();
      for (vector<Card> c1 : playerPossibleCards) {
        for (vector<Card> c2 : boardPossibleCards) {
          combined[0] = c1[0];
          combined[1] = c1[1];
          combined[2] = c2[0];
          combined[3] = c2[1];
          combined[4] = c2[2];
          score = max(score, getScoreOfHand(combined));
        }
      }
      if (type == "NLH") {
        // Four on board, one in players hand 
        vector<vector<Card>> playerPossibleCards = player.generatePossibleOneCard();
        vector<vector<Card>> boardPossibleCards = board.generatePossibleFourCards();
        for (vector<Card> c1 : playerPossibleCards) {
          for (vector<Card> c2 : boardPossibleCards) {
            combined[0] = c1[0];
            combined[1] = c2[0];
            combined[2] = c2[1];
            combined[3] = c2[2];
            combined[4] = c2[3];
            score = max(score, getScoreOfHand(combined));
          }
        }

        // Playing the board
        combined[0] = board.cards[0];
        combined[1] = board.cards[1];
        combined[2] = board.cards[2];
        combined[3] = board.cards[3];
        combined[4] = board.cards[4];
        score = max(score, getScoreOfHand(combined));
      }

      return score;
    }

    void addBoard(string s) {
      Board newBoard = Board(s);
      boards.push_back(newBoard);
    }

    void addPlayer(string s) {
      Player newPlayer = Player(s);
      players.push_back(newPlayer);
    }

    void setType(string s) {
      type = s;
    }

    string getType() {
      return type;
    }

    void setNumBoards(int val) {
      if (val <= 0) {
        throwError("Illegal number of boards!");
      }
      numBoards = val;
    }

    int getNumBoards() {
      return numBoards;
    }

    void setNumPlayers(int val) {
      numPlayers = val;
    }

    int getNumPlayers() {
      return numPlayers;
    }

    void printBoards() {
      for (Board board : boards) {
        board.printBoard();
      }
    }

    void printPlayers() {
      for (Player player: players) {
        player.printPlayer();
      }
    }

};

int main( int argc, char *argv[] ) {
  if (argc == 1) {
    throwError("Please provide an input text file!");
  } 
  string fileName = argv[1];
  ifstream infile (fileName);
  if (!infile.good()) {
    throwError("Input text file invalid!");
  }
  
  Showdown showDown;
  showDown.parseInput(fileName);
  showDown.calculateScores();
  showDown.calculatePayouts();
  showDown.printPayouts();

}