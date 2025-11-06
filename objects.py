import cupy as cp
from util import normalize

class light():
    def __init__(self, direction):
        self.direction = normalize(cp.array(direction, dtype = cp.float32))

    def rays(self):
            return self.direction
            
class sphere:
    def __init__(self, centre, radius, color):

        self.centre = cp.array(centre, dtype = cp.float32)
        self.radius = radius
        self.color = cp.array(color, dtype = cp.uint8)

    def distance(self, point):
        # return the shortest length from the sphere surface to the camera
        return cp.linalg.norm(point - self.centre, axis = -1) - self.radius

    def normal(self, point):
        # Steve's logic for gradient
        # epsilon set to 0.01
        ex = cp.array([0.01, 0, 0])
        ey = cp.array([0, 0.01, 0])
        ez = cp.array([0, 0, 0.01])
        
        dx = cp.linalg.norm(point + ex, axis = -1) - cp.linalg.norm(point, axis = -1)
        dy = cp.linalg.norm(point + ey, axis = -1) - cp.linalg.norm(point, axis = -1)
        dz = cp.linalg.norm(point + ez, axis = -1) - cp.linalg.norm(point, axis = -1)

        gradient = cp.stack((dx, dy, dz), axis=-1)

        return normalize(gradient)
    
    def hitpixels(self, eye, steps, epsilon):
        rays = eye.rays()
        pixels = cp.broadcast_to(eye.pos, rays.shape).copy()
    
        hitbox = cp.zeros((eye.height, eye.width), dtype = cp.float32)

        for i in range(steps):
            d = self.distance(pixels)

            hit = d < epsilon
            hitbox = (hitbox + hit) > 0

            d = cp.where(hit, 0, d)
            pixels = pixels + rays * d[:, :, None]

        return self.normal(pixels[hitbox]), hitbox


