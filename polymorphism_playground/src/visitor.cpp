#include <iostream>

#include "visitor.hpp"

void Visitor::visit(ProgramNode& node) {
    std::cout << "Visiting ProgramNode" << std::endl;
    node.fn.accept(*this);
}

void Visitor::visit(ImportNode& node) {
    std::cout << "Visiting FieldNode" << std::endl;
}

