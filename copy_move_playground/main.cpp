// reading a text file
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

class StringContainer {
  private:
    string* ptr;
  public:
    // Constructor
    StringContainer (const string& str) : ptr(new string(str)) {
      cout << "Normal Constructor!" << endl;;
    }

    // Destructor
    ~StringContainer () { 
      cout << "Destructor" << endl;
      delete ptr; 
    }

    // Copy Constructor
    StringContainer (const StringContainer& other) : ptr(new string(other.getContent())) {
      cout << "Copy Constructor!" << endl;;
    }

    // Copy Assignment - note that generally we can delete the current pointer and allocate space
    // We can also use the copy-and-swap idiom where we pass in by value and then swap the values
    StringContainer& operator= (const StringContainer& other) {
      cout << "Copy Assignment!" << endl;;
      *ptr = other.getContent();
      return *this;
    }

    // Move Constructor
    StringContainer (StringContainer&& other) : ptr(other.ptr) {
      cout << "Move Constructor!" << endl;;
      other.ptr = nullptr;
    }

    // Move Assignment - in this case, we must clear out the original data
    StringContainer& operator= (StringContainer&& other) {
      cout << "Move Assignment!" << endl;;
      delete ptr;
      ptr = other.ptr;
      other.ptr = nullptr;
      return *this;
    }

    // Accessor Function
    const string& getContent() const {
      return *ptr;
    }

    // Example Operation
    StringContainer operator+ (const StringContainer& rhs) {
      StringContainer ret = StringContainer(this->getContent() + " " + rhs.getContent());
      return ret;
    }

    int getStringLength() const;

    void printContents() {
      cout << getContent() << "\n\n";
    }
};

int StringContainer::getStringLength() const {
  return (*ptr).length();
}

StringContainer generateStringContainer(string input) {
  // SimpleArray ret = SimpleArray(size);
  // return ret;

  return StringContainer(input);
}

int main () {
  StringContainer first = StringContainer("hi"); // Normal hi
  first.printContents();

  StringContainer second = StringContainer("hello"); // Normal hello
  second.printContents();

  StringContainer third = first; // Copy constructor hi
  third.printContents();

  StringContainer fourth = generateStringContainer("hey there"); // Normal, Move Constructor hey there
  fourth.printContents();

  third = generateStringContainer("heyo!"); // Move assignment heyo!
  third.printContents();

  second = first + third; // Move assignment hi heyo!
  second.printContents();
}