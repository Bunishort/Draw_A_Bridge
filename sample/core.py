def conv2(matrix, kernel):
    '''
    :param matrix: 2D numpy matrix
    :param kernel: 2D numpy matrix, smaller than matrix in each direction
    :return: conv : convoluted matrix
    '''
    import numpy as np
    conv = np.convolve(matrix,kernel)
    return conv

def CG_loop(solid,Axx,Axy,Ayy, Ux,Uy)