import random
import numpy as np
cimport numpy as np
# cimport rand_int from random_x

cdef int[:,:] directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype='i')

ctypedef fused char_or_int:
    int
    char

cdef bint in_bounds(char_or_int[:,:] g, int y, int x):
    return x >= 0 and y >= 0 and y < g.shape[0] and x < g.shape[1]

cdef bint is_frontier(int[:, :] img, int i, int j):
    cdef int k, di, dj
    for k in range(directions.shape[0]):
        di = i + directions[k, 0]
        dj = j + directions[k, 1]
        if in_bounds(img, di, dj) and not img[di, dj]:
            return True
    return False

cpdef tuple mutate(int[:, :] img, int n_mutations):
    cdef int i, j, k, di, dj, ai, aj, ri, rj
    cdef list options
    cdef bint is_front
    cdef set front = set()
    cdef tuple p1, p_rem, p3
    img = img.copy()
    
    # Create the frontier.
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if not img[i, j]:
                continue
            if is_frontier(img, i, j):
                front.add((i, j))
            
    for _ in range(n_mutations):
        # Add one and remove one to keep count constant.
        p1, p_rem = random.sample(tuple(front), 2)
        
        # Remove one and add its neighbors to front.
        img[p_rem[0], p_rem[1]] = 0
        front.remove(p_rem)
        for k in range(directions.shape[0]):
            di = p_rem[0] + directions[k, 0]
            dj = p_rem[1] + directions[k, 1]
            if in_bounds(img, di, dj) and img[di, dj]:
                front.add((di, dj))

        # Add a neighbor from i.
        options = []
        for k in range(directions.shape[0]):
            di = p1[0] + directions[k, 0]
            dj = p1[1] + directions[k, 1]
            if in_bounds(img, di, dj) and not img[di, dj]:
                options.append((di, dj))

        if len(options):
            p3 = random.choice(options)
            img[p3[0], p3[1]] = 1
            front.add(p3)

        # Adding p3 may make p1 an interior.
        if not is_frontier(img, p1[0], p1[1]):
            front.remove(p1)

    return np.asarray(img), front 


# cpdef np.ndarray repair(int[:, :] img, float target, float tol, int max_iters=1000):
#     cdef int i, j, k, di, dj, ai, aj, ri, rj
#     cdef list options
#     cdef bint is_front
#     cdef set front = set()
#     cdef tuple p1, p_rem, p3
#     img = img.copy()
    
#     cdef int count = np.sum(img)
    
#     # Create the frontier.
#     for i in range(1, img.shape[0]-1):
#         for j in range(1, img.shape[1]-1):
#             if not img[i, j]:
#                 continue
#             if is_frontier(img, i, j):
#                 front.add((i, j))

#     for _ in range(max_iters):
#         p1 = random.choice(tuple(front))
        
#         if count > target * img.size:
#             # Remove one and add its neighbors to front.
#             img[p1[0], p1[1]] = 0
#             front.remove(p1)
#             count -= 1
#             for k in range(directions.shape[0]):
#                 di = p1[0] + directions[k, 0]
#                 dj = p1[1] + directions[k, 1]
#                 if in_bounds(img, di, dj) and img[di, dj]:
#                     front.add((di, dj))
#         else:
#             # Add a neighbor from i.
#             options = []
#             for k in range(directions.shape[0]):
#                 di = p1[0] + directions[k, 0]
#                 dj = p1[1] + directions[k, 1]
#                 if in_bounds(img, di, dj) and not img[di, dj]:
#                     options.append((di, dj))

#             if len(options):
#                 p3 = random.choice(options)
#                 img[p3[0], p3[1]] = 1
#                 count += 1
#                 front.add(p3)

#             # Adding p3 may make p1 an interior.
#             if not is_frontier(img, p1[0], p1[1]):
#                 front.remove(p1)
        
#         if abs(count - target * img.size) < tol:
#             break

#     return np.asarray(img)