// reading a text file
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

void print_file(string file_name) {
    string curr_line;
    ifstream myfile (file_name);
    if (myfile.is_open()) {
        while (getline(myfile, curr_line)) {
            cout << curr_line << "\n";
        }
        myfile.close();
    } else {
        cout << "Unable to open file!";
    }
}

int get_file_size(string file_name) {
    ifstream myfile (file_name, ios::binary);

    streampos begin = myfile.tellg();

    myfile.seekg (0, ios::end);
    streampos end = myfile.tellg();

    myfile.close();
    return end - begin;
}

int main () {
    string curr_file = "text_file.txt";
    print_file(curr_file);
    cout << to_string(get_file_size(curr_file)) << endl;
    
    ofstream myfile;
    myfile.open(curr_file, std::ios_base::app);
    if (myfile.is_open()) {
        string curr_line;
        while (true) {
            getline(cin, curr_line);
            if (curr_line == "exit") {
                cout << "Exiting program";
                myfile.close();
                break;
            } else {
                myfile << curr_line << "\n";
            }
        }
    } else {
        cout << "Failed to open file to write to";
    }
}