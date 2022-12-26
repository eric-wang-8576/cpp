import subprocess
import time
import random
import argparse
import glob
import os

class InteractiveBinary():
    def __init__(self, path_to_bin):
        self.bin_path = path_to_bin

        output_file = f'tmpout_{random.randint(0,1000)}'
        self.output_writer = open(output_file, "wb")
        self.output_reader = open(output_file, "r")
        self.proc = subprocess.Popen([f'./{path_to_bin}'], stdin=subprocess.PIPE, stdout=self.output_writer, stderr=self.output_writer, bufsize=1, universal_newlines=True)

        time.sleep(0.1) # give binary enough time for engine to setup

    def run_uci_command(self, command):
        command = command.strip("\n") + "\n"
        self.proc.stdin.write(command)
        time.sleep(1) # give time for output to be written to tmpout
        return self.output_reader.read()

    def close(self):
        self.proc.kill()

class TestSuite():
    def __init__(self, current_binary, num_trials):
        self.current_binary = current_binary
        self.num_trials = num_trials

    def remove_tmp_files(self):
        for tmp_file in glob.glob('tmpout_*'):
            os.remove(tmp_file)

    def run_tests(self):
        self.run_correctness()
        self.remove_tmp_files()
    
    def extract_value(self, result_string):
        return int(result_string.split(" ")[3])

    def run_correctness(self):
        print("Running Correctness Tests:")
        current_binary_uci = InteractiveBinary(self.current_binary)

        for i in range(self.num_trials):
            val1 = random.randint(0, 1000)
            val2 = random.randint(0, 1000)
            print("Test " + str(i + 1) + ": " + str(val1) + " and " + str(val2))
            expected_sum = val1 + val2

            current_binary_uci.run_uci_command(str(val1))
            returned_string = current_binary_uci.run_uci_command(str(val2))
            returned_sum = self.extract_value(returned_string)

            if expected_sum == returned_sum:
                print("Pass!")
            else:
                print("Fail!")
                print("Correct value is " + str(expected_sum))
                print("Your computed value is " + str(returned_sum))
            print("\n")

        current_binary_uci.close()

def test_driver(current_binary, num_trials):
    test_suite = TestSuite(current_binary, num_trials)
    test_suite.run_tests()

parser = argparse.ArgumentParser()
parser.add_argument('--current-binary', type=str, default='chatbot')
parser.add_argument('--num-trials', type=int, default='chatbot')
args = parser.parse_args()

if __name__ == "__main__":
    test_driver(args.current_binary, args.num_trials)