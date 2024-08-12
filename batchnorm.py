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

    if (tensor.abs().max() > tensor.max()):
        scale = (-qmin)/tensor.abs().max()
    else:
        scale = (qmax)/tensor.max()
    
    if mode == 'unsigned':
        tensor_q = (tensor*scale).round().clamp(min=0, max=2**(num_bits)-1).to(torch.int32)
    else:
        tensor_q = (tensor*scale).round().clamp(min=-2**(num_bits-1), max=2**(num_bits-1)-1).to(torch.int32)

    return tensor_q, scale

def to_binary(x, bits):
    if x == -2**(bits-1):
        x = x + 1
    if x < 0:
        # Compute two's complement for negative numbers
        x = (1 << (bits)) + x
    elif x >= (1 << bits):
        # Handle overflow for unsigned values
        x = x % (1 << bits)
    return format(x, f'0{bits}b')

def binary_to_bits(arr, axis=1):
    if arr.ndim == 1:
        # Handle 1D case
        num_elements = arr.shape[0]
        bit_length = len(arr[0])
        result = np.zeros((num_elements * bit_length,), dtype=int)
        for i in range(num_elements):
            bits = list(map(int, arr[i]))
            result[i * bit_length:(i + 1) * bit_length] = bits
    elif arr.ndim == 2:
        # Handle 2D case
        num_rows, num_cols = arr.shape
        bit_length = len(arr[0, 0])
        
        if axis == 1:  # Expands horizontally
            result = np.zeros((num_rows, num_cols * bit_length), dtype=int)
            for i in range(num_rows):
                for j in range(num_cols):
                    bits = list(map(int, arr[i, j]))
                    result[i, j * bit_length:(j + 1) * bit_length] = bits
        elif axis == 0:  # Expands vertically
            result = np.zeros((num_rows * bit_length, num_cols), dtype=int)
            for i in range(num_rows):
                for j in range(num_cols):
                    bits = list(map(int, arr[i, j]))
                    result[i * bit_length:(i + 1) * bit_length, j] = bits
        else:
            raise ValueError("axis must be either 0 (bits in rows) or 1 (bits in columns)")
    else:
        raise ValueError("Input array must be either 1D or 2D")
    
    return result

vectorized_to_binary = np.vectorize(to_binary)

# Load input and weight tensors
inp_tensor = torch.load('conv1_input.pth')
weight_tensor = torch.load('conv1_weights.pth')

stride = 2
padding = 3

N, C, H, W = inp_tensor.shape
F, _, filter_h, filter_w = weight_tensor.shape

out_h = (H + 2 * padding - filter_h) // stride + 1
out_w = (W + 2 * padding - filter_w) // stride + 1

print("Max value of input tensor:",inp_tensor.max().item()) # To find out if 1 can be incorporated to perform Batch Norm

x_bits = 6
w_bits = 9

Rc = torch.load('Rc_resnet50_cifar10.pth')
Tc = torch.load('Tc_resnet50_cifar10.pth')

# Quantising W_in and Rc together
comb_w = torch.cat((weight_tensor.flatten(),Rc.flatten()))
_, scale_w_Rc = quantize_tensor(comb_w, w_bits, mode='signed')

w_new_q = (weight_tensor*scale_w_Rc.cpu().detach()).round().clamp(min=-2**(w_bits-1), max=2**(w_bits-1)-1).to(torch.int32)
weight_binary_batchnorm = vectorized_to_binary(w_new_q, w_bits)

Rc_q = (Rc*scale_w_Rc.cpu().detach()).round().clamp(min=-2**(w_bits-1), max=2**(w_bits-1)-1).to(torch.int32)

Rc_np = Rc_q.detach().cpu().numpy()
# print("RC_decimal", Rc_np)

Rc_bin = vectorized_to_binary(Rc_np, w_bits)

def binary_to_twos_complement(binary_array):
    
    int_array = np.array([int(b, 2) for b in binary_array])
    bit_length = len(binary_array[0])

    subbed_array = np.array([2**bit_length - i for i in int_array])
    
    twos_complement_array = np.array([to_binary(num, bit_length)[:(bit_length)] for num in subbed_array])
    
    return twos_complement_array

Rc_2s_comp_binstr = binary_to_twos_complement(Rc_bin)
# print("2's comp RC", Rc_2s_comp)

# Use weight_binary_batchnorm and Rc_2s_comp_binstr to combine and send into conv2d_batchnorm to be broken into bit columns

# Adding column of 1s to input tensor
ones = torch.ones((2,2))

comb_in = torch.cat((inp_tensor.flatten(), ones.flatten()))
_, scale_x_ones = quantize_tensor(comb_in, x_bits, mode='signed')

x_new_q = (inp_tensor*scale_x_ones.cpu().detach()).round().clamp(min=-2**(x_bits-1), max=2**(x_bits-1)-1).to(torch.int32)
inp_binary_batchnorm = vectorized_to_binary(x_new_q, x_bits)

inp_last_col_element = (ones*scale_x_ones.cpu().detach()).round().clamp(min=-2**(x_bits-1), max=2**(x_bits-1)-1).to(torch.int32)

col_length = N*out_h*out_w

ones_col_array = np.full(col_length, inp_last_col_element[0][0].item())
ones_col_array_binstr = vectorized_to_binary(ones_col_array, x_bits)

# Use inp_binary_batchnorm and ones_col_array_binstr to combine and send into conv2d_batchnorm to be broken into bit rows

def add_1d_to_2d(array_2d, array_1d, axis):
    
    array_2d = np.array(array_2d)
    array_1d = np.array(array_1d)
    
    if axis == 0:  # Adding as a row
        if array_1d.shape[0] != array_2d.shape[1]:
            raise ValueError("The length of the 1D array must match the number of columns in the 2D array.")
        result = np.vstack([array_2d, array_1d])
        
    elif axis == 1:  # Adding as a column
        if array_1d.shape[0] != array_2d.shape[0]:
            raise ValueError("The length of the 1D array must match the number of rows in the 2D array.")
        result = np.hstack([array_2d, array_1d[:, np.newaxis]])
        
    else:
        raise ValueError("Axis must be 0 (add as row) or 1 (add as column).")
    
    return result

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

def conv2d_batchnorm(input, filters, bias=None, stride=1, padding=0):
    N, C, H, W = input.shape
    F, _, filter_h, filter_w = filters.shape

    # Calculate output dimensions
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    # Reshape input and filters
    col = im2col(input, filter_h, filter_w, x_bits, w_bits, stride, padding) # Change number of bits of input quantization 
    filters_col = filters.reshape(F, -1).T

    col_bin_mod = add_1d_to_2d(col, ones_col_array_binstr, axis=1)
    filters_bin_mod = add_1d_to_2d(filters_col, Rc_2s_comp_binstr, axis=0)

    col_bin = binary_to_bits(col_bin_mod, axis=0)
    filters_bin = binary_to_bits(filters_bin_mod, axis=1)

    # print(col_bin.shape)
    # print(filters_bin.shape)

    # Matrix multiplication for Convolution + Batch Norm
    out = np.dot(col_bin, filters_bin)

    if bias is not None:
        out += bias

    # Reshape the output
    # out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

    return out

batchnorm_out = conv2d_batchnorm(inp_binary_batchnorm, weight_binary_batchnorm, bias=None, stride=2, padding=3)

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

res_bn, subarray_res_bn = break_array_to_subarrays(batchnorm_out, x_bits, w_bits)

bn_out_np = res_bn.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)
bn_out_np = bn_out_np/(scale_w_Rc*scale_x_ones).item()

print("Final batchnorm output shape:",bn_out_np.shape)

bn_out_tensor = (torch.from_numpy(bn_out_np))*(Tc[None, :, None, None].cpu().detach())
print(bn_out_tensor[0][0][3])

print(scale_w_Rc)