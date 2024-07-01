#include "util.hpp"

std::string valueToString(int value) {
    bool negative = value < 0;
    std::string str = std::to_string(std::abs(value));
    std::string res;

    int n = str.length() - 1;
    for (int i = n; i >= 0; i--) {
        if ((n - i) % 3 == 0 && i != n) {
            res.insert(res.begin(), ',');
        }
        res.insert(res.begin(), str[i]);
    }

    if (negative) {
        res.insert(res.begin(), '-');
    }

    return res;
}

std::string priceToString(int price) {
    bool negative = price < 0;
    std::string str = std::to_string(std::abs(price));
    std::string res;

    int n = str.length() - 1;
    for (int i = n; i >= 0; i--) {
        if ((n - i) % 3 == 0 && i != n) {
            res.insert(res.begin(), ',');
        }
        res.insert(res.begin(), str[i]);
    }

    res.insert(res.begin(), '$');
    if (negative) {
        res.insert(res.begin(), '-');
    }

    return res;
}

std::string PNLToString(int PNL) {
    bool negative = PNL < 0;
    std::string str = std::to_string(std::abs(PNL));
    std::string res;

    int n = str.length() - 1;
    for (int i = n; i >= 0; i--) {
        if ((n - i) % 3 == 0 && i != n) {
            res.insert(res.begin(), ',');
        }
        res.insert(res.begin(), str[i]);
    }

    res.insert(res.begin(), '$');
    if (negative) {
        res.insert(res.begin(), '-');
    } else {
        res.insert(res.begin(), '+');
    }

    return res;
}
