// lower_bound/upper_bound example
#include <iostream>     // std::cout
#include <algorithm>    // std::lower_bound, std::upper_bound, std::sort
#include <vector>       // std::vector

int main () {
  int myints[] = {10,20,30,30,20,10,10,20};
  std::vector<int> v(myints,myints+8);           // 10 20 30 30 20 10 10 20

  std::sort (v.begin(), v.end());                // 10 10 10 20 20 20 30 30

  int tests[] = {9, 10, 15, 20, 25, 30, 35};

  for (int i = 0; i < 7; i++) {
    int val = std::lower_bound(v.begin(), v.end(), tests[i]) - v.begin();
    std::cout << "lower bound for position " << std::to_string(tests[i]) << " is " << std::to_string(val) << std::endl;
  }

  std::cout << std::endl;

  for (int i = 0; i < 7; i++) {
    int val = std::upper_bound(v.begin(), v.end(), tests[i]) - v.begin();
    std::cout << "upper bound for position " << std::to_string(tests[i]) << " is " << std::to_string(val) << std::endl;
  }

  return 0;
}