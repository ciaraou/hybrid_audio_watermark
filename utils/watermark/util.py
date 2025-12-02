

def tobits(*, s: str) -> list[int]:
    """ Takes string and converts each ascii character to binary in list of int form """
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def frombits(*, bits: list[int]) -> str:
    """ Take list of ints and combine into ascii string, reverses tobits """
    chars = []
    for b in range(len(bits) / 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def visual_bitwise_check(*, arr1: str, arr2: str) -> str:
    """ takes two bitstrings and compares them, 0 or 1 where they're the same, * where they're different. """
    if len(arr1) != len(arr2):
        print("visual_bitwise_check requires equal length")
        return -1
    
    return "".join(["*" if arr1[i] != arr2[i] else arr2[i] for i in range(len(arr2))])

