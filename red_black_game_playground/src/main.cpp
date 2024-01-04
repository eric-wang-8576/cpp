#include "game.hpp"
#include "strategy.hpp"

#define NUM_TRIALS 10000

void simulate(void (*strategy) ()) {
}

int main() {
    // Simulate Random
    double avg;
    double min;
    double max;

    avg = 0;
    min = 10000;
    max = -10000;
    for (int i = 0; i < NUM_TRIALS; ++i) {
        Game game {GameUtil::minBet, GameUtil::maxBet};

        Strategy strategy {&game};
        strategy.random();

        int pnl = game.getPnl();
        min = pnl < min ? pnl : min;
        max = pnl > max ? pnl : max;
        avg += pnl;
    }
    avg /= NUM_TRIALS;
    std::cout << "RANDOM STRATEGY:" << std::endl;
    std::cout << "Average across " << NUM_TRIALS << " trials is " << avg << std::endl;
    std::cout << "Max PNL: " << max << std::endl;
    std::cout << "Min PNL: " << min << std::endl;
    std::cout << std::endl;

    // Simulate Strategy
    avg = 0;
    min = 10000;
    max = -10000;
    for (int i = 0; i < NUM_TRIALS; ++i) {
        Game game {GameUtil::minBet, GameUtil::maxBet};

        Strategy strategy {&game};
        strategy.play();

        int pnl = game.getPnl();
        min = pnl < min ? pnl : min;
        max = pnl > max ? pnl : max;
        avg += pnl;
    }
    avg /= NUM_TRIALS;
    std::cout << "TARGETED STRATEGY:" << std::endl;
    std::cout << "Average across " << NUM_TRIALS << " trials is " << avg << std::endl;
    std::cout << "Max PNL: " << max << std::endl;
    std::cout << "Min PNL: " << min << std::endl;
    std::cout << std::endl;
}
