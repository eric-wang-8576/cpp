/*
 * Protected members are accessible from other members of the same class but also from
 *   members of their derived classes
 *
 * If we declare a member function within a class, the function is automatically considered
 *   an inline member function by the compiler and can enable optimizations
 *
 * Classes allow programming using object-oriented paradigms - data and functions are both 
 *   members of the object, reducing the need to pass and carry handlers or other state
 *   variables as arguments to functions, because they are part of the object whose member is called
 *
 * Empty parentheses cannot be used to call the default constructor, which is a constructor that takes
 *   no parameters
 *
 * Constructors with a single parameter can be called using variable initialization syntax
 *
 * For member objects (those whose type is a class), if they are not initialized after the colon,
 *   they are default-constructed. If there is no default constructor, members are initialized in 
 *   the member initialization list
 */

#include <iostream>

// RECTANGLE

class Rectangle {
    int width, height;
public:
    Rectangle();
    Rectangle(int, int);
    void set_values(int, int);
    int area() { return width * height; }
};

Rectangle::Rectangle() {
    width = 0;
    height = 0;
}

Rectangle::Rectangle(int a, int b) {
    width = a;
    height = b;
}

void Rectangle::set_values(int x, int y) {
    width = x;
    height = y;
}

// CIRCLE

class Circle {
    double radius;
public:
    Circle(double r) { radius = r; }
    double area() { return radius * radius * 3.14159265; }
};

class Cylinder {
    Circle base;
    double height;
public:
    Cylinder(double r, double h) : base{r}, height(h) {}
    double volume() { return base.area() * height; }
};

int main() {
    Rectangle rect, rectb;
    rect.set_values(3, 4);
    rectb.set_values(5, 6);
    std::cout << "rect area: " << rect.area() << std::endl;
    std::cout << "rectb area: " << rectb.area() << std::endl;

    Circle foo(10.0);
    Circle bar = 20.0;
    Circle baz {30.0};

    Cylinder cylinder {10, 20};
    
    std::cout << "cylinder's volume: " << cylinder.volume() << std::endl;

    return 0;
}

