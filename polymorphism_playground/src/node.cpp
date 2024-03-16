#include "node.hpp"

void Node::accept(Visitor& visitor) {
    visitor.accept(*this);
}
