import numpy as np

def flatten_upper_triangle(matrix):
    upper_triangle=np.triu(matrix)
    flat=upper_triangle.flatten()
    non_zero=flat[flat > 0]
    return non_zero