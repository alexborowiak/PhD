import numpy as np
import pandas as pd
import xarray as xr
import sys

input_number = sys.argv[1]

def test_function_1(a,b):
    c = a **b 
    print(c)
    return c


def main():
    
    test_function_1(2,1)
    test_function_1(2,2)
    test_function_1(2,3)
    test_function_1(2,4)
    
    print(f'{input_number=}')
    print(type(input_number))
    
    
if __name__ == '__main__':
    main()