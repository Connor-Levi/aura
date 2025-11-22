import cupy as cp

def normalize(v):
    length = cp.linalg.norm(v, axis = -1, keepdims = True)

    # in case length is zero
    length = cp.where(length == 0, 1, length)
    
    return v / length

def blackhole(img, bx, by, s):
    # strength of distortion
    strength = s * 1000

    # image copy
    warped = cp.empty_like(img)
    h, w, _ = img.shape

    # calculating new coordinates
    x, y = cp.meshgrid(cp.arange(w), cp.arange(h))

    # offset
    dx = x - bx
    dy = y - by

    r = cp.sqrt(dx**2 + dy**2)

    distort = strength / r

    # unit vectors
    i = dx / r
    j = dy / r

    # new coordinates
    x1 = x - distort * i
    y1 = y - distort * j

    # keeping it in range
    x1 = cp.clip(x1,0, w-1).astype(cp.int32)
    y1 = cp.clip(y1,0, h-1).astype(cp.int32)

    warped[:, :, :] = img[y1, x1]

    # creating an event horizon
    horizon = 75

    black = r < horizon
    warped[black] = cp.array([0, 0, 0], dtype=img.dtype)

    return warped