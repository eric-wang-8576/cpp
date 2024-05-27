#include <random>
#include <iostream>

#define NUMVALS 10000000

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(1, 5);

int gen1to5() {
    return dis(gen);
}

int gen1to7() {
    int val;
    do {
        val = (gen1to5() - 1) * 5 + gen1to5() - 1;
    } while (val >= 21);

    return (val % 7) + 1;
}

int main() {
    std::vector<int> distribution (7, 0);
    for (int i = 0; i < NUMVALS; ++i) {
        distribution[gen1to7() - 1]++;
    }

    for (auto val : distribution) {
        std::cout << val << " ";
    }
}
