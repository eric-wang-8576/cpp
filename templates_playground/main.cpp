#include <iostream>
using namespace std;

template <class T>
class mypair {
  T a;
  T b;
  public:
    mypair (T first, T second) {
      a = first;
      b = second;
    }
    
    T getmax();
};

template <class T>
T mypair<T>::getmax() {
  T ret;
  ret = a > b ? a : b;
  return ret;
}

template <class T>
T product(T a, T b) {
  T res;
  res = a*b;
  return res;
}

template <class T, int N>
T multiply_by_int(T a) {
  T res;
  res = a * N;
  return res;
}


int main () {
  mypair<float> newObj (40.6, 302.390);
  cout << to_string(newObj.getmax()) << endl;
  cout << product<int>(3, 5) << endl;
  cout << multiply_by_int<float, 3>(1.25) << endl;
  return 0;
}