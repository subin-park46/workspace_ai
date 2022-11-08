import numpy as np

def celsius_to_fahrenheit(x):
    return x * 1.8 + 32

data_c = np.array(range(0, 100))
data_f = celsius_to_fahrenheit(data_c)
print(data_f)