import sys
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def read_file(file_path):
    with open(file_path, 'r') as file:
        contents = file.readlines()
    return contents

def convert_data(contents):
    return [int(value.strip().replace('$', '').replace(',', '')) for value in contents]

def get_all_percentile_values(arr):
    """
    Sorts a giant array of integers and returns the values at each percentile from 0 to 99.
    
    :param arr: List or array of integers
    :return: Dictionary with percentile as key and corresponding value as value
    """
    sorted_arr = np.sort(arr)
    percentiles = range(100)
    percentile_values = {percentile: np.percentile(sorted_arr, percentile) for percentile in percentiles}
    return percentile_values

def plot_histogram(data, bins=50, title='PNL Distributions', xlabel='PNL', ylabel='Frequency'):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_histogram.py <file_name>")
        sys.exit(1)

    file_path = sys.argv[1]
    contents = read_file(file_path)
    title = contents[0]
    data = convert_data(contents[1:])
    percentile_values = get_all_percentile_values(data)

    # Preparing data for a more compact display
    compact_data = []
    for i in range(20):
        compact_data.append([])
    for percentile, value in percentile_values.items():
        sign = "+$" if value >= 0 else "-$"
        formatted_value = f"{sign}{abs(value):,.2f}"
        compact_data[(percentile) % 20].append(f"{percentile}%: {formatted_value}")

    # Printing the compact table
    print(tabulate(compact_data, tablefmt="plain"))
    plot_histogram(data, 50, title, 'PNL', 'Frequency')
