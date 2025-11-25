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

    # new coordinates grid
    x, y = cp.meshgrid(cp.arange(w), cp.arange(h))

    # offset
    dx = x - bx
    dy = y - by

    # radial distance
    r = cp.sqrt(dx ** 2 + dy ** 2)

    # unit vectors
    i = dx / r
    j = dy / r

    # distortion factor
    distort = strength / r

    # new coordinates
    x1 = x - distort * i
    y1 = y - distort * j

    # keeping the new coordinates in range of the image size
    x1 = cp.clip(x1,0, w-1).astype(cp.int32)
    y1 = cp.clip(y1,0, h-1).astype(cp.int32)

    # updating the warped image
    warped[:, :, :] = img[y1, x1]

    # set horizon > 0 to see enable event horizon around the centre of the black hole
    horizon = 0

    # if distance is less than the horizon, colors the pixels black
    black = r < horizon
    warped[black] = cp.array([0, 0, 0], dtype=img.dtype)

    return warped