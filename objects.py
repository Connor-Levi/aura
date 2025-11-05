import cupy as cp

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

