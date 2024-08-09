import numpy as np
import torch
import torch.nn.functional as F

def quantize_tensor(tensor, num_bits, mode='signed'):

    if mode == 'unsigned':
        qmin = 0
        qmax = 2**num_bits - 1
    else:
        qmin = -2**(num_bits - 1)
        qmax = 2**(num_bits - 1) - 1

    if (tensor.abs().max() > tensor.max()) and (tensor.abs().max() > 1):
        scale = (-qmin)/tensor.abs().max()
    else:
        scale = (qmax)/tensor.max()
    
    if mode == 'unsigned':
        tensor_q = (tensor*scale).round().clamp(min=0, max=2**(num_bits)-1).to(torch.int32)
    else:
        tensor_q = (tensor*scale).round().clamp(min=-2**(num_bits-1), max=2**(num_bits-1)-1).to(torch.int32)

    return tensor_q, scale

def to_binary(x, bits):
        if x < 0:
            # Compute two's complement for negative numbers
            x = (1 << (bits)) + x
        return format(x, f'0{bits}b')

def binary_to_bits(arr, axis=1):
    # Calculate the total number of bits
    num_rows, num_cols = arr.shape
    bit_length = len(arr[0, 0])
    
    if axis == 1:
        # Place bits in new columns
        result = np.zeros((num_rows, num_cols * bit_length), dtype=int)
        for i in range(num_rows):
            for j in range(num_cols):
                bits = list(map(int, arr[i, j]))
                result[i, j * bit_length:(j + 1) * bit_length] = bits

    elif axis == 0:
        # Place bits in new rows
        result = np.zeros((num_rows * bit_length, num_cols), dtype=int)
        for i in range(num_rows):
            for j in range(num_cols):
                bits = list(map(int, arr[i, j]))
                result[i * bit_length:(i + 1) * bit_length, j] = bits
    else:
        raise ValueError("axis must be either 0 (bits in rows) or 1 (bits in columns)")
    
    return result

vectorized_to_binary = np.vectorize(to_binary)

# Load weight and input tensors
weight_tensor = torch.load('conv1_weights.pth')
inp_tensor = torch.load('conv1_input.pth')

print("Max value of input tensor:",inp_tensor.max().item()) # To find out if 1 can be incorporated to perform Batch Norm

# print(weight_tensor.shape, inp_tensor.shape)

x_bits = 6
w_bits = 8

w_q, scale_w = quantize_tensor(weight_tensor, w_bits)
x_q, scale_x = quantize_tensor(inp_tensor, x_bits)

input = x_q.cpu().detach().numpy()  # (N, C, H, W)
filters = w_q.cpu().detach().numpy()  # (F, C, filter_h, filter_w)

input_binary = vectorized_to_binary(input, 6)
weight_binary = vectorized_to_binary(filters, 8)

def im2col(input, filter_h, filter_w, x_bits, w_bits, stride=1, padding=0):
    N, C, H, W = input.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(input, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant',constant_values='0'*x_bits)
    col = np.full((N, C, filter_h, filter_w, out_h, out_w),'0'*x_bits)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def conv2d(input, filters, bias=None, stride=1, padding=0):
    N, C, H, W = input.shape
    F, _, filter_h, filter_w = filters.shape

    # Calculate output dimensions
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    # Reshape input and filters
    col = im2col(input, filter_h, filter_w, x_bits, w_bits, stride, padding) # Change number of bits of input quantization 
    filters_col = filters.reshape(F, -1).T

    col_bin = binary_to_bits(col, axis=0)
    filters_bin = binary_to_bits(filters_col, axis=1)

    # print(col_bin.shape)
    # print(filters_bin.shape)
    
    # Perform matrix multiplication
    out = np.dot(col_bin, filters_bin)
    if bias is not None:
        out += bias

    # Reshape the output
    # out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
    conv_dims = (N, out_h, out_w, F)
    return out, conv_dims

output, conv_dims = conv2d(input_binary, weight_binary, bias=None, stride=2, padding=3)
# print(output,output.shape,np.max(output))

def break_array_to_subarrays(array, x, y):
    # Check if the array can be evenly divided into sub-arrays of shape (x, y)
    assert array.shape[0] % x == 0, "The array cannot be evenly divided along the x dimension."
    assert array.shape[1] % y == 0, "The array cannot be evenly divided along the y dimension."
    
    # Calculate the number of sub-arrays in each dimension
    num_sub_arrays_x = array.shape[0] // x
    num_sub_arrays_y = array.shape[1] // y
    
    # Initialize an empty object array to hold the condensed values
    condensed_values = np.empty((num_sub_arrays_x, num_sub_arrays_y))
    subarrays = np.empty((num_sub_arrays_x, num_sub_arrays_y, x, y), dtype=array.dtype)
    
    # Loop through the array and process sub-arrays
    for i in range(num_sub_arrays_x):
        for j in range(num_sub_arrays_y):
            sub_array = array[i*x:(i+1)*x, j*y:(j+1)*y]
            subarrays[i, j] = sub_array  # Store the sub-array
            condensed_value = 0
            for sub_i in range(sub_array.shape[0]):
                for sub_j in range(sub_array.shape[1]):
                    if (sub_i==0 and sub_j!=0) or (sub_i!=0 and sub_j==0):
                        condensed_value -= sub_array[sub_i, sub_j] * 2**((x + y - 2) - (sub_i + sub_j))
                    else:
                        condensed_value += sub_array[sub_i, sub_j] * 2**((x + y - 2) - (sub_i + sub_j))

            condensed_values[i, j] = condensed_value
    
    return condensed_values, subarrays

res, subarray_res = break_array_to_subarrays(output, x_bits, w_bits)
# print(res.shape)

N, out_h, out_w, F = conv_dims
conv_out_np = res.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
conv_out_np = conv_out_np/(scale_w*scale_x).item()

print("Final conv output shape:",conv_out_np.shape)
print(conv_out_np[0][0][3])

conv_out_tens = torch.from_numpy(conv_out_np)
# print(conv_out_tens)

#### END OF CONVOLUTION ####

#### START OF BATCH NORM ####

subarray_reshaped = subarray_res.reshape(N, out_h, out_w, F, x_bits, w_bits).transpose(0, 3, 1, 2, 4, 5)
print("Subarray with MACs before 2's power mult shape",subarray_reshaped.shape)
# print(subarray_reshaped[0][0][2][5])