#include <algorithm>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

class SimpleArray {
  private:
    size_t arraySize;
    int* arrayP;

    void swap(SimpleArray& first, SimpleArray& second) {
      using std::swap;
      swap(first.arraySize, second.arraySize);
      swap(first.arrayP, second.arrayP);
    }

    // // ANOTHER POSSIBLE IMPLEMENTATION USING MOVE:
    // void swap(SimpleArray& a, SimpleArray& b) {
    //   SimpleArray tmp (std::move(a));
    //   a = std::move(b);   
    //   b = std::move(tmp);
    // }

  public:
    // Constructor
    SimpleArray (size_t size = 0) : arraySize(size), arrayP(arraySize ? new int[arraySize] : nullptr) {
      cout << "Normal Constructor! ";
    }

    // Destructor
    ~SimpleArray () {
      delete [] arrayP;
    }

    // Copy-constructor
    SimpleArray (const SimpleArray& other) : arraySize(other.arraySize), arrayP(arraySize ? new int[arraySize] : nullptr) {
      cout << "Copy Constructor! ";
      copy(other.arrayP, other.arrayP + other.arraySize, arrayP);
    }

    // Copy-assignment
    // Importantly, passing in by value allows the compiler to make a copy efficiently
    // The way that other is initialized can be either copy or move constructor
    SimpleArray& operator= (SimpleArray other) {
      cout << "Copy Assignment! ";
      swap(*this, other);
      return *this;
    }

    // Move-constructor
    // What we do is initialize via the default constructor, then swap with other
    SimpleArray (SimpleArray&& other) : SimpleArray() {
      cout << "Move constructor! ";
      swap(*this, other);
    }

    void printSize() const {
      cout << to_string(arraySize) << endl;
    }
};

SimpleArray generateSimpleArray(int size) {
  // SimpleArray ret = SimpleArray(size);
  // return ret;

  return SimpleArray(size);
}

int main () {
  // PRINT POINTERS!
  SimpleArray p1 = SimpleArray(5); // Normal, 5
  p1.printSize();

  SimpleArray p2 = SimpleArray(8); // Normal, 8
  p2.printSize();

  SimpleArray p3 = p1; // Copy, 5
  p3.printSize();

  p2 = p1; // Copy Assignment
  p2.printSize();

  SimpleArray p4 = generateSimpleArray(3); // Normal, Move Constructor, 3
  p4.printSize();

  p3 = generateSimpleArray(13); // Normal, Move Assignment, 13
  p3.printSize();

  cout << endl;
  return 0;
}