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
  public:
    SimpleArray (size_t size = 0) : arraySize(size), arrayP(arraySize ? new int[arraySize] : nullptr) {
      cout << "Normal Constructor! ";
    }

    // Copy-constructor
    SimpleArray (const SimpleArray& other) : arraySize(other.arraySize), arrayP(arraySize ? new int[arraySize] : nullptr) {
      cout << "Copy Constructor! ";
      copy(other.arrayP, other.arrayP + other.arraySize, arrayP);
    }

    ~SimpleArray () {
      delete [] arrayP;
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
      swap(*this, other);
    }

    void printSize() const {
      cout << to_string(arraySize) << endl;
    }
};

SimpleArray generateSimpleArray(int size) {
  SimpleArray ret = SimpleArray(size);
  return ret;
}

int main () {
  SimpleArray p1 = SimpleArray(5);
  p1.printSize();
  SimpleArray p2 = SimpleArray(8);
  p2.printSize();
  SimpleArray p3 = p1;
  p3.printSize();
  p2 = p1;
  p2.printSize();
  SimpleArray p4 = generateSimpleArray(3);
  p4.printSize();
  return 0;
}