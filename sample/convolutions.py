import numpy as np

def addition_convolution(matrix, kernel):
    """Performs a convolution using only additions and subtractions."""

    h,w = matrix.shape
    hs = np.arange(h-2) + 1
    ws = np.arange(w-2) + 1
    kh, kw = kernel.shape
    result = np.zeros_like(matrix)
    khs = np.arange(kh)
    kkhs = np.arange(kh) - (kh - 1) // 2
    kws = np.arange(kw)
    kkws = np.arange(kw) - (kw - 1) // 2

    # No computations on the edge of the matrix to simplify. Only works with 9*9 kernels max
    for i in hs:
        for j in ws:
            for ki,kki in zip(khs,kkhs):
                for kj,kkj in zip(kws,kkws):
                    if kernel[ki, kj] == 1:  # Addition
                        result[i, j] += matrix[(i+kki) ,(j+kkj) ]
                    elif kernel[ki, kj] == -1:  # Subtraction
                        result[i, j] -= matrix[(i+kki) ,(j+kkj) ]

    return result
