import cupy as cp

def normalize(v):
    length = cp.linalg.norm(v, axis = -1, keepdims = True)

    # in case length is zero
    length = cp.where(length == 0, 1, length)
    
    return v / length
