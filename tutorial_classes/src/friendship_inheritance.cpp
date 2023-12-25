/*
 * A non-member function can access the private and protected members of a class if it 
 *   is declared a friend of that class - it must be declared inside the class - typical
 *   use cases of friend functions are operations that are conducted between two different
 *   classes accessing private or protected members of both
 *
 * A class can access the private and protected members of another class if it is declared
 *   a friend within that class
 *
 * The access specifier of a class limits the most accessible level for members inherited 
 *   from the base class 
 *
 * The protected access specifier means that when a class inherits another one, the members
 *   of the derived class can access the protected members inherited from the base class
 *
 * A publicly derived class inherits access to every member of a base class except for
 *   its constructors and destructor, its assignment operator members, its friends, and its
 *   private members
 *
 * The constructors of a derived class calls the default constructor of its base classes,
 *   and we can specify a different constructor if we want
 */

#include <iostream>

// Typically would go in a header file
class Rectangle;
class Square;

class Rectangle {
    int width, height;
public:
    Rectangle() {}
    Rectangle(int x, int y) : width(x), height(y) {}
    int area() { return width * height; }
    friend Rectangle duplicate(const Rectangle&);
    void convert(Square&);
};

class Square {
    friend class Rectangle;
    int side;
public:
    Square(int a) : side(a) {}
};

void Rectangle::convert(Square& sq) {
    width = sq.side;
    height = sq.side;
}

Rectangle duplicate(const Rectangle& param) {
    Rectangle res;
    res.width = param.width * 2;
    res.height = param.height * 2;
    return res;
}


class Mother {
public:
    Mother() {
        std::cout << "Mother: no parameters" << std::endl;
    }
    Mother(int a) {
        std::cout << "Mother: int parameter" << std::endl;
    }
};

class Daughter : public Mother {
public:
    Daughter(int a) {
        std::cout << "Daughter: int parameter" << std::endl;
    }
};

class Son : public Mother {
public:
    Son(int a) : Mother(a) {
        std::cout << "Son: int parameter" << std::endl;
    }
};

// A class can inherit from output to call T::print(i)
class Output {
public:
    static void print(int i);
};

void Output::print(int i) {
    std::cout << i << std::endl;
}

int main() {
    Rectangle foo;
    Rectangle bar {2, 3};
    foo = duplicate(bar);
    std::cout << foo.area() << std::endl;

    Square sq {5};
    foo.convert(sq);
    std::cout << foo.area() << std::endl;

    Daughter kelly {0};
    Son bob {0};

}

