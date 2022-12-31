// reading a text file
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
using namespace std;

#define MAX_HOLE_CARDS 4
#define NUM_BOARD_CARDS 5

vector<string> split(const string &s, char delim) {
  vector<string> result;
  stringstream ss (s);
  string item;

  while (getline (ss, item, delim)) {
      result.push_back (item);
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
}

int getSuitRank(string s) {
  if (s == "c" || s == "C") { return 0; }
  else if (s == "d" || s == "D") { return 1; }
  else if (s == "h" || s == "H") { return 2; }
  else if (s == "s" || s == "S") { return 3; }
}

bool isStraight(int* valFreq) {
  // Wheel
  if (valFreq[12] == 1) {
    for (int i = 0; i < 4; i++) {
      if (valFreq[i] != 1) {
        break;
      }
    }
    return true;
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

int getScoreOfHand(vector<Card> combinedCards) {
  int ret;
  int* valFreq = new int[13];
  int* suitFreq = new int[4]; 
  for (Card currCard : combinedCards) {
    int length = currCard.size();
    valFreq[getCardRank(currCard.substr(0, length - 1))]++;
    suitFreq[getSuitRank(currCard.substr(length - 1, 1))]++;
  }

  // CONTINUE IMPLEMENTING HERE
}

typedef string Card;

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
      cout << "Generating Possible Cards For Board:" << endl;
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
};

class Player {
  public:
    string name;
    double stackSize;
    HoleCards holeCards;
    int highestScore;
    
    int payout;
    Player (string s) {
      vector<string> vec = split(s, ' ');
      name = vec[0];
      stackSize = stod(vec[1]); 

      holeCards = HoleCards();
      for (int i = 2; i < vec.size(); i++) {
        holeCards.cards[i - 2] = vec[i];
      }
      holeCards.numCards = vec.size() - 2;

      highestScore = 0;
    }

    void printPlayer() {
      cout << "Player name is: " << name << endl;
      cout << "Player highest score is: " << highestScore << endl;
      cout << "Player stack size is: " << fixed << setprecision(2) << stackSize << endl;
      holeCards.printHoleCards();
      cout << "\n";

      // PRINT GENERATED CARDS
      vector<vector<Card>> generatedCards = generatePossibleTwoCards();
      cout << "Generating Possible Cards For Player:" << endl;
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
};

class Showdown {
  private:
    int numPlayers;
    int numBoards;
    string type;

  public:
    vector<Player> players;
    vector<Board> boards;

    Showdown() {
      numPlayers = 0;
      numBoards = 0;
      players = vector<Player>();
      boards = vector<Board>();
    }

    int calculatePlayerBoardScore(Player player, Board board) {
      if (type == "PLO") {
        int score = 0;
        vector<vector<Card>> playerPossibleCards = player.generatePossibleTwoCards();
        vector<vector<Card>> boardPossibleCards = board.generatePossibleThreeCards();
        for (vector<Card> playerCards : playerPossibleCards) {
          for (vector<Card> boardCards : boardPossibleCards) {
            vector<Card> combinedCards (playerCards);
            combinedCards.insert(playerCards.end(), boardCards.begin(), boardCards.end());
            score = max(score, getScoreOfHand(combinedCards));
          }
        }
      }
      return 0;
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

int main () {
  Showdown showDown;
  
  string fileName = "input.txt";
  string currLine;
  ifstream myfile (fileName);

  // capture type

  getline(myfile, currLine);
  showDown.setType(currLine);

  // capture boards

  getline(myfile, currLine);
  showDown.setNumBoards(stoi(currLine));

  for (int i = 0; i < showDown.getNumBoards(); i++) {
    getline(myfile, currLine);
    showDown.addBoard(currLine);
  }

  showDown.printBoards();

  // capture players

  getline(myfile, currLine);
  showDown.setNumPlayers(stoi(currLine));

  for (int i = 0; i < showDown.getNumPlayers(); i++) {
    getline(myfile, currLine);
    showDown.addPlayer(currLine);
  }

  showDown.printPlayers();





  // if (myfile.is_open()) {
  //     while (getline(myfile, currLine)) {
  //         cout << currLine << "\n";
  //     }
  //     myfile.close();
  // } else {
  //     cout << "Unable to open file!";
  // }

    
  //   print_file(currFile);
}