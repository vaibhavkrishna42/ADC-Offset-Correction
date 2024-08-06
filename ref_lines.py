import csv
import itertools
import argparse

def load_last_n_rows_into_columns(filename, n):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Get the last n rows
    last_n_rows = rows[-n:]

    # Convert the values to floats
    last_n_rows = [[float(value) for value in row] for row in last_n_rows]

    # Transpose rows to columns
    columns = list(map(list, zip(*last_n_rows)))

    return columns

def find_closest_sum(numbers, target):
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

    return closest_numbers, closest_sum

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

def write_sub_to_file(sub, filename):
    with open(filename, 'w') as file:
        file.write(str(sub))

def main(data_file, indices_output_file, rows, target_num):
    RL_columns = load_last_n_rows_into_columns(data_file, rows)

    flat_list = [item for sublist in RL_columns for item in sublist]

    closest_numbers, sub = find_closest_sum(flat_list, target_num)

    print(f"Closest sum: {sub}")

    write_sub_to_file(sub, 'sub.txt')

    index_list = []
    for i in range(len(RL_columns)):
        index_list.append([])

    for i in closest_numbers:
        for j in range(len(RL_columns)):
            if i in RL_columns[j]:
                temp_index = RL_columns[j].index(i)
                index_list[j].append((temp_index + 1021))  # Adjust if number of rows changes

    write_nested_list_to_csv_columns(index_list, indices_output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple CSV files with a specified number of rows.")
    parser.add_argument('--data_file', type=str, required=True, help='The data filename')
    parser.add_argument('--indices_output_file', type=str, required=True, help='The output filename for indices')
    parser.add_argument('--rows', type=int, required=True, help='The number of rows to load')
    parser.add_argument('--target_num', type=float, required=True, help='The target number to find the closest sum to')

    args = parser.parse_args()

    main(args.data_file, args.indices_output_file, args.rows, args.target_num)