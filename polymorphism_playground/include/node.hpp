#pragma once

class Visitor;

class Node {
public:
    virtual void accept(Visitor& visitor);
};


class ImportNode : public Node {
};

class ProgramNode : public Node {
public:
    ImportNode fn;
};
