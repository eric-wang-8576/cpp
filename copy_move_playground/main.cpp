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
    StringContainer (const string& str) : ptr(new string(str)) {}

    // Destructor
    ~StringContainer () { 
      delete ptr; 
    }

    // Copy Constructor
    StringContainer (const StringContainer& other) : ptr(new string(other.getContent())) {
      cout << "Copy Constructor! ";
    }

    // Copy Assignment - note that generally we can delete the current pointer and allocate space
    // We can also use the copy-and-swap idiom where we pass in by value and then swap the values
    StringContainer& operator= (const StringContainer& other) {
      cout << "Copy Assignment! ";
      *ptr = other.getContent();
      return *this;
    }

    // Move Constructor
    StringContainer (StringContainer&& other) : ptr(other.ptr) {
      cout << "Move Constructor! ";
      other.ptr = nullptr;
    }

    // Move Assignment - in this case, we must clear out the original data
    StringContainer& operator= (StringContainer&& other) {
      cout << "Move Assignment! ";
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
};

int StringContainer::getStringLength() const {
  return (*ptr).length();
}

int main () {
  StringContainer first = StringContainer("hi");
  StringContainer second = StringContainer("hello");
  first = first + second;
  cout << first.getContent() << endl;
  cout << first.getStringLength() << endl;
}