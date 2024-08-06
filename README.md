# Terminal code to run ref_lines.py
python3 ref_lines.py --data_file RL_4p5k_1.csv --indices_output_file RL_4p5k_1_4r_10s_index.csv --rows 4 --target_num 10

# Terminal code to run data_lines.py
python3 data_lines.py --offset_file DL_1_offset.csv --dl_file DL_4p5k_1.csv --indices_output_file DL_4p5k_1_15r_10s_index.csv --error_output_file DL_4p5k_1_15r_10s_error.csv --rows 15

# Replace 4p5k with 9k if different resistor value is used.
# Replace 1 with 2 when sub-array 2 is used.

# 10s in data_lines.py command is to indicate that 10mV is subtracted to remove positive offset so change appropriately if different value is used.

# Offset file must contain original offsets (in mV) without any shifting