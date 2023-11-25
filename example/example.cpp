#include <iostream>
#include <string>
using namespace std;

class MyClass {
    int x;
  public:
    MyClass(int val) : x(val) {}
    const int& get() const {return x;}
    int& get() {return x;}
};

class Example3 {
    string data;
  public:
    Example3 (const string& str) : data(str) {}
    Example3() {}
    const string& content() const {return data;}
};

int main() {
  MyClass foo (10);
  const MyClass bar (20);
  foo.get() = 15;         // ok: get() returns int&
  cout << foo.get() << '\n';
  cout << bar.get() << '\n';
  cout << __cplusplus << endl;

  Example3 foo2;
  Example3 bar2 ("Example");

  cout << "foo2's content: " << foo2.content() << '\n';
  cout << "bar2's content: " << bar2.content() << '\n';
  return 0;
}
