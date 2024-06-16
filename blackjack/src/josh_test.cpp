#include "engine/shoe.hpp"

int main() {
    std::vector<int> vals(11);

    Shoe shoe {6};

    for (int i = 0; i < 1000000; ++i) {
        Card* cardP = shoe.draw();
        vals[cardP->getVal()]++;
        shoe.triggerShuffle();
    }

    for (int i = 1; i <= 11; ++i) {
        std::cout << "Vals with " << i << " is " << vals[i] << std::endl;
    }
}
