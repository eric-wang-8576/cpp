/*
 * Polymorphism is the art of taking advantage of the simple but power and versatile 
 *   feature that a pointer to a derived class is type-compatible with a pointer to its
 *   base class - a polymorphic class is one that declares or inherits a virtual member
 *
 * A virtual member is a member function that can be redefined in a derived class, while
 *   preserving its calling properties through references - non-virtual members can be
 *   redefined in derived classes, but they cannot be accessed through a reference of the
 *   base class 
 *
 * Abstract base classes are only used as base classes and have pure virtual functions - 
 *   they cannot be used to instantiate objects, but can have pointers
 */

#include <iostream>

class Polygon {
protected:
    int width, height;
public:
    Polygon(int a, int b) : width(a), height(b) {}
    virtual int area() const =0; // Pure Virtual Function
    // Note: Move semantics potentially disabled
    virtual ~Polygon() {}
    void printArea() {
        std::cout << this->area() << std::endl;
    }
};

class Rectangle : public Polygon {
public:
    Rectangle(int a, int b) : Polygon {a, b} {}
    ~Rectangle() {}
    int area() const {
        return width * height;
    }
};

class Triangle : public Polygon {
public:
    Triangle(int a, int b) : Polygon {a, b} {}
    ~Triangle() {}
    int area() const {
        return width * height / 2;
    }
};

int main() {
    Polygon* p1 = new Rectangle{4, 5};
    Polygon* p2 = new Triangle{4, 5};
    p1->printArea();
    p2->printArea();
    delete p1;
    delete p2;
}
