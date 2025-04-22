import numpy as np
import struct

# Endian standard: BIG
FLOAT64_BITSIZE = 64

# util functions
def even_audio(x_ea):
    """ 
    Makes audio buffer even length so it can be 2N long accurately. 
    
    Parameters
    ----------
    x_ea : array-like
        audio buffer

    Returns
    -------
    array-like
        audio buffer (even length)
    """
    
    if len(x_ea) % 2 != 0:
        x_ea = np.append(x_ea, [0])
    return x_ea

def float64_to_uint64_bits(value): # type punning/ bit-level reinterpretation of signed float to uint
    return struct.unpack('>Q', struct.pack('>d', value))[0]

def uint64_bits_to_float64(bits): # type punning/ bit-level reinterpretation of uint to signed float
    return struct.unpack('>d', struct.pack('>Q', bits))[0]

def float64_to_binary(num):
    return ''.join(f'{byte:08b}' for byte in struct.pack('>d', num))

def bytestring_to_bitstring(byte_string):
    return ''.join(format(byte, '08b') for byte in byte_string)

def bit_change(value, bit, index=0):
    mask = (1 << (index % FLOAT64_BITSIZE)) # system endian doesn't matter because float64_to_uint64_bits does big endian

    value_int = float64_to_uint64_bits(value) # convert to uint64 (type only - binary same) for bitwise op
    
    if bit == "1":
        res = value_int | mask # bitwise or: value at index will be 1 
    else:
        # print("zero?")
        # print(bit)
        res = value_int & ~mask # bitwise and not: value at index will be 0
        # print(res)
        # print("")

    return uint64_bits_to_float64(res) # convert back to float64 type

def visual_bitwise_check(arr1, arr2):
    if len(arr1) != len(arr2):
        print("visual_bitwise_check requires equal length")
        return -1
    
    return "".join(["*" if arr1[i] != arr2[i] else arr2[i] for i in range(len(arr2))])

def zero_lsb(signal, index=0):
    res = np.copy(signal)
    for i in range(len(signal)):
        res[i] = bit_change(signal[i], 0, index)
    return res

# todo for watermark check
# def check_binary(wmbits):
    # binary_int = int(wmbits, 2)
    # byte_number = binary_int.bit_length() + 7 // 8
    # binary_array = binary_int.to_bytes(byte_number, "big") 
    # ascii_text = binary_array.decode()
    # print(ascii_text)