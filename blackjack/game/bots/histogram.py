import sys
import matplotlib.pyplot as plt

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [int(value.strip().replace('$', '').replace(',', '')) for value in data]

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
    data = read_data(file_path)
    plot_histogram(data)
