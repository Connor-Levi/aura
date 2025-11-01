import cupy as cp

class sphere:
    def __init__(self, centre, radius, color):

        self.centre = cp.array(centre, dtype = cp.float32)
        self.radius = radius
        self.color = cp.array(color, dtype = cp.uint8)

    def distance(self, point):
        # return the shortest length from the sphere surface to the camera
        return cp.linalg.norm(point - self.centre, axis = -1) - self.radius
