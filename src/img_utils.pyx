import numpy as np
cimport numpy as np

cpdef np.ndarray subdivide(int[:,:] img):
    cdef int i, j, v, ii, jj
    cdef int[:,:] result = np.zeros((img.shape[0]*2, img.shape[1]*2), dtype='i')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            v = img[i, j]
            ii = 2*i
            jj = 2*j
            result[ii, jj] = v
            result[ii+1, jj] = v
            result[ii, jj+1] = v
            result[ii+1, jj+1] = v
    return np.asarray(result)