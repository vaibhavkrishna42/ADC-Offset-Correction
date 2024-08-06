import argparse
import itertools
import numpy as np
import csv

def load_sub_from_file(filename):
    """Load the float value from a file."""
    with open(filename, 'r') as file:
        return float(file.read().strip())

def find_indices(numbers, subset, rows):
    indices = []
    for num in subset:
        try:
            index = numbers.index(num) + 1025 - rows   # To get row number
            indices.append((index))
        except ValueError:
            # Handle case where number not found in the list
            print(f"Number {num} not found in the original list.")
    return indices

def find_closest_sum(numbers, target, rows):
    def subset_sums(numbers):
        sums = {}
        for r in range(len(numbers) + 1):
            for subset in itertools.combinations(numbers, r):
                s = sum(subset)
                if s not in sums:
                    sums[s] = subset
        return sums

    n = len(numbers)
    left_half = numbers[:n // 2]
    right_half = numbers[n // 2:]

    left_sums = subset_sums(left_half)
    right_sums = subset_sums(right_half)

    closest_sum = float('inf')
    closest_numbers = None

    for left_sum in left_sums:
        for right_sum in right_sums:
            current_sum = left_sum + right_sum
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum
                closest_numbers = left_sums[left_sum] + right_sums[right_sum]

    indices = find_indices(numbers, closest_numbers, rows)
    error = closest_sum - target

    return indices, error

def load_last_n_rows_into_columns(filename, row_num):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    ###########################################################################
    # Get the last n rows
    last_n_rows = rows[-row_num:]
    ###########################################################################

    # Convert the values to floats
    last_n_rows = [[float(value) for value in row] for row in last_n_rows]

    # Transpose rows to columns
    columns = list(map(list, zip(*last_n_rows)))

    return columns

def load_single_row_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        row = next(reader)  # Read the first row

    sub = load_sub_from_file('sub.txt')

    # Convert the elements to floats
    row = [np.abs(float(value) - sub) for value in row]  # Positive offset correction by value found from ref_lines.py
    return row

def write_nested_list_to_csv_columns(nested_list, filename, placeholder=None):
    # Find the maximum length of the lists
    max_length = max(len(lst) for lst in nested_list)

    # Pad shorter lists with the placeholder
    padded_list = [lst + [placeholder] * (max_length - len(lst)) for lst in nested_list]

    # Transpose the nested list to convert rows to columns
    transposed_list = list(map(list, zip(*padded_list)))

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in transposed_list:
            writer.writerow(row)

def write_list_to_single_row_csv(data_list, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_list)

def main(filename_target, filename_DL, filename_indices, filename_error, rows):

    targets = load_single_row_csv(filename_target)
    columns = load_last_n_rows_into_columns(filename_DL, rows)

    indices = []
    error = []

    for i in range(len(columns)):
        p = find_closest_sum(columns[i], targets[i], rows)
        indices.append(p[0])
        error.append(p[1])

    print(max(error), min(error))
    # print(indices)
    # print(len(indices))

    write_nested_list_to_csv_columns(indices, filename_indices)
    write_list_to_single_row_csv(error, filename_error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process target and DL files.")
    parser.add_argument('--offset_file', type=str, required=True, help='The target offset filename')
    parser.add_argument('--dl_file', type=str, required=True, help='The DL filename')
    parser.add_argument('--indices_output_file', type=str, required=True, help='The output filename for indices')
    parser.add_argument('--error_output_file', type=str, required=True, help='The output filename for error')
    parser.add_argument('--rows', type=int, required=True, help='The number of rows to load')

    args = parser.parse_args()

    main(args.offset_file, args.dl_file, args.indices_output_file, args.error_output_file, args.rows)
