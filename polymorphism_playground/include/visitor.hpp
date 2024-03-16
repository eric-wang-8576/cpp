#pragma once

#include "node.hpp"

class Visitor {
public:
    void visit(ProgramNode&);
    void visit(ImportNode&);
};
