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
    }
};

class Showdown {
  private:
    int numPlayers;
    int numBoards;

  public:
    vector<Player> players;
    vector<Board> boards;

    Showdown() {
      numPlayers = 0;
      numBoards = 0;
      players = vector<Player>();
      boards = vector<Board>();
    }

    void addBoard(string s) {
      Board newBoard = Board(s);
      boards.push_back(newBoard);
    }

    void addPlayer(string s) {
      Player newPlayer = Player(s);
      players.push_back(newPlayer);
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