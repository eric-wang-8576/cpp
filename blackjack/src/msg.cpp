#include "msg.hpp"

std::string actionBorder {"~~~~~~~~~~~~~~~~~~~~~~~"};

void Msg::print() {
    std::cout << std::endl;
    std::cout << actionBorder << std::endl;
    std::cout << prevActionConfirmation << std::endl;
    std::cout << actionBorder << std::endl;
    std::cout << std::endl;
}
