#include <iostream>
using namespace std;

class Polygon {
  protected:
    int width, height;
  public:
    Polygon (int a, int b) : width(a), height(b) {}

    // Virtual function
    virtual int area (void) =0;
    
    void printArea() {
      cout << this->area() << endl;
    }
};

class Rectangle: public Polygon {
  public:
    Rectangle (int a, int b) : Polygon(a, b) {}
    int area() {
      return width * height;
    }

    int area(int factor) {
      return width * height * factor;
    }
};

class Triangle: public Polygon {
  public:
    Triangle(int a, int b) : Polygon(a, b) {}
    int area() {
      return width * height / 2;
    }
};

int main () {
  Rectangle rect1 = Rectangle(4, 5);
  Polygon* p1 = &rect1;
  Polygon* p2 = new Triangle(3, 6);

  p1->printArea();
  p2->printArea();
  cout << rect1.area(3) << "\n";

  return 0;
}