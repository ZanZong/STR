from ctypes import cdll

# great_module = cdll.LoadLibrary('./_swigCall.so')
# a = great_module._wrap_Max(1, 3)
# print(a)
import test_module
print(test_module.get_max(1,3))