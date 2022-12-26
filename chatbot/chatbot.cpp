// echo machine
#include <iostream>
#include <string>
#include <exception>
using namespace std;

int main ()
{
  string str;
  unsigned int status = 0;
  int val = 0;
  int num_vals = 2;

  #ifdef USETHREE 
    num_vals = 3;
  #endif


  while (true) {
    cout << "Enter an integer: ";
    getline (cin,str);

    try {
      val += std::stoi(str);
    } catch (exception& e) {
      if (str == "exit") {
        break;
      } else {
        cout << "Please enter a valid integer." << endl;
        continue;
      }
    }
    
    status++;
    if (status == num_vals) {
        cout << "Your sum is " << val << " " << endl << "\n";
        status = 0;
        val = 0;
    }
  }
}