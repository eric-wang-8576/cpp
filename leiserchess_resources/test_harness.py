import subprocess
import time
import random
import argparse
import glob
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser()
parser.add_argument('--test-binary', type=str, default='leiserchess')
parser.add_argument('--original-binary', type=str, default='bin/5_better_options')
args = parser.parse_args()


class InteractiveBinary():
    def __init__(self, path_to_bin):
        self.bin_path = path_to_bin
        output_file = f'tmpout_{random.randint(0,1000)}'
        self.output_writer = open(output_file, "wb")
        self.output_reader = open(output_file, "r")
        self.proc = subprocess.Popen([f'./{path_to_bin}'], stdin=subprocess.PIPE, stdout=self.output_writer, stderr=self.output_writer, bufsize=1, universal_newlines=True)
        time.sleep(0.1) # allow binary enough time for engine to setup

    # runs UCI command, returns output
    def run_uci_command(self, command):
        command = command.strip("\n") + "\n"
        self.proc.stdin.write(command)
        time.sleep(1.8) # ensure that enough time has passed for the output to be written to tmpout
        return self.output_reader.read()

    def run_eval(self):
        return self.run_uci_command("eval\n")

    def close(self):
        self.proc.kill()

class TestSuite():
    def __init__(self, original_binary, test_binary, test_list, num_trials = 5):
        self.original_binary = original_binary
        self.test_binary = test_binary
        self.test_list = test_list
        self.num_trials = num_trials

    def print_test_header(self, test_name):
        pad_length = max(46, len(test_name))
        header_str = "-" * (2 * pad_length + len(test_name) + 2)
        middle_str = "-" + " " * pad_length + test_name + " " * pad_length + "-"
        print(bcolors.BOLD + header_str + '\n' + middle_str + '\n' + header_str + bcolors.ENDC)

    def remove_tmps(self):
        for tmp_file in glob.glob('tmpout_*'):
            os.remove(tmp_file)

    def fetch_keys(self, test_key_gen):
        msg = "Generating keys"
        print(f"{msg:.<100}", end='')
        keys = test_key_gen.generate_keys(self.num_trials)
        print("DONE")
        return keys
    
    def run_test(self, test_name, test, test_key_gen):
        self.print_test_header(test_name)
        test_keys = self.fetch_keys(test_key_gen)
        for key in test_keys:
            original_result, test_result = test(self.original_binary, self.test_binary, key)
            print(f"{key:.<100}", end='')
            if original_result == test_result:
                print(bcolors.OKGREEN + "PASS" + bcolors.ENDC)
            else:
                print(bcolors.FAIL + "FAIL" + bcolors.ENDC)
                print(f"{self.original_binary}:\n {original_result}")
                print(f"{self.test_binary}:\n {test_result}")
        self.remove_tmps()
            
    def run_tests(self):
        for (test_name, test, test_key_gen) in self.test_list:
            self.run_test(test_name, test, test_key_gen)

###############################################
###              KEY GENERATION             ###
###############################################

class KeyGenerator():
    def __init__(self, original_binary, key_gen_fn):
        self.original_binary = original_binary
        self.key_gen_fn = key_gen_fn

    def generate_keys(self, num_keys):
        key_set = set()
        while len(key_set) < num_keys:
            new_key = self.key_gen_fn(self.original_binary)
            key_set.add(new_key)
        return list(key_set)

def generate_fen_key(original_binary):
    original_binary_uci = InteractiveBinary(original_binary)
    fen_string = " ". join(original_binary_uci.run_uci_command("randomboard\n").split()[-2:])
    original_binary_uci.close()
    return fen_string

fen_key_generator = KeyGenerator(args.original_binary, generate_fen_key)

###############################################
###                  TESTS                  ###
###############################################


# filter to include only lines of the following two forms:
#   TOTAL [HEURISTIC] contribution of [VALUE] for [COLOR]
#   info score cp [SCORE]
def extract_heuristic_values(eval_result):
    eval_lines = eval_result.split('\n')
    filtered_lines = list(filter(lambda line: line[:5] in ("Total", "info "), eval_lines))
    return "\n".join(filtered_lines)


def test_eval(original_binary, test_binary, fen_key):
    original_binary_uci = InteractiveBinary(original_binary)
    test_binary_uci = InteractiveBinary(test_binary)

    # set both uci boards to new  fen string
    original_binary_uci.run_uci_command(f"position fen {fen_key}\n")
    test_binary_uci.run_uci_command(f"position fen {fen_key}\n")
    
    # evaluate both on fen string
    original_result = extract_heuristic_values(original_binary_uci.run_eval())
    test_result = extract_heuristic_values(test_binary_uci.run_eval())

    # kill processes
    original_binary_uci.close()
    test_binary_uci.close()

    # return test key, expected result, actual result
    return original_result, test_result

def test_perft(original_binary, test_binary, fen_key):
    original_binary_uci = InteractiveBinary(original_binary)
    test_binary_uci = InteractiveBinary(test_binary)

    # set both uci boards to new  fen string
    original_binary_uci.run_uci_command(f"position fen {fen_key}\n")
    test_binary_uci.run_uci_command(f"position fen {fen_key}\n")

    # obtain perft values from both ucis
    original_perft = original_binary_uci.run_uci_command("perft 6")
    test_perft = test_binary_uci.run_uci_command("perft 6")

    # kill processes
    original_binary_uci.close()
    test_binary_uci.close()

    return original_perft, test_perft  

# each entry of test list must be of the form
# ("TEST_NAME", test_fn, key_generator)
test_list = [
    ("TEST_PERFT", test_perft, fen_key_generator),
    ("TEST_EVAL", test_eval, fen_key_generator),
]

def test_driver(original_binary, test_binary):
    test_suite = TestSuite(original_binary, test_binary, test_list)
    test_suite.run_tests()

if __name__ == "__main__":
    test_driver(args.original_binary, args.test_binary)