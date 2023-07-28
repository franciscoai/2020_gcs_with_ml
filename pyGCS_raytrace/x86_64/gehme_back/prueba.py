import ctypes
import array
printf = libc.printf
printf.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
printf(b"String '%s', Int %d, Double %f\n", b"Hi", 10, 2.2)