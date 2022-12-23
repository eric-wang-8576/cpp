// echo machine
#include <iostream>
#include <string>
#include <exception>
using namespace std;


class Polygon {
  protected:
    int width, height;
  public:
    Polygon (int a, int b) : width(a), height(b) {}
};

class Output {
  public:
    static void print (int i);
};

void Output::print (int i) {
  cout << i << '\n';
}

class Rectangle: public Polygon, public Output {
  public:
    Rectangle (int a, int b) : Polygon(a,b) {}
    int area ()
      { return width*height; }
};

class Triangle: public Polygon, public Output {
  public:
    Triangle (int a, int b) : Polygon(a,b) {}
    int area ()
      { return width*height/2; }
};

int main ()
{
  string str;
  unsigned int status = 0;
  int val = 0;
  int num_vals = 2;

  #ifdef USETHREE 
    num_vals = 3;
  #endif


  while (true) {
    cout << "Enter an integer: ";
    getline (cin,str);

    try {
      val += std::stoi(str);
    } catch (exception& e) {
      if (str == "exit") {
        break;
      } else {
        cout << "Please enter a valid integer." << endl;
        continue;
      }
    }
    
    status++;
    if (status == num_vals) {
        cout << "Your sum is " << val << endl << "\n";
        status = 0;
        val = 0;
    }
  }
}