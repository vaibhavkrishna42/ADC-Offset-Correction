import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram_from_csv(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, header=None)  # No column names

    # Flatten the DataFrame to a single list of values
    data = df.values.flatten()

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins='auto', edgecolor='black')
    plt.title('Histogram of CSV Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Add text for mean and standard deviation
    textstr = f'Mean: {mean:.2f}\nStd Dev: {std_dev:.2f}'
    # Use matplotlib's text function to place the text box
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.5))

    # Show the plot
    plt.show()

# Usage example
csv_file = r'C:\Users\USER\VSCode\ADCoffset\Error_Data\DL_4p5k_1_15r_10s_error.csv' # Corrected offsets
# csv_file = r'C:\Users\USER\VSCode\ADCoffset\DL_1_offset.csv' # Original offsets

plot_histogram_from_csv(csv_file)
