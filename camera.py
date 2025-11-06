import cupy as cp
from util import normalize

class camera:
    def __init__(self, pos, point, up, fov, width, height):
        """
        pos: position of the camera
        point: position where the camera points
        up: vertical up
        fov: field of view in degrees
        width, height: width and height of the image
        p_width, p_height: define dimensions of the image in actual units
        """

        self.pos = cp.array(pos, dtype = cp.float32)
        self.point = cp.array(point, dtype = cp.float32)
        self.up = cp.array(up, dtype = cp.float32)
        self.fov = cp.radians(fov)
        self.width = width
        self.height = height

        # unit vectors originating from the camera
        self.forward = normalize(self.point - self.pos)
        self.right = normalize(cp.cross(self.forward, self.up))
        self.cam_up = normalize(cp.cross(self.right, self.forward))
        
        ratio = width / height

        self.p_width = 2 * cp.tan(self.fov / 2)
        self.p_height = self.p_width / ratio

    def rays(self):
        # making a grid
        i, j = cp.meshgrid(cp.arange(self.width), cp.arange(self.height))
        
        # magnitudes of components of ray vectors
        u = self.p_width * (i / (self.width - 1) - 0.5)
        v = self.p_height * (0.5 - j / (self.height - 1))
    
        ray_vec = (self.forward[None, None, :] + 
                u[:, :, None] * self.right[None, None, :] +
                v[:, :, None] * self.cam_up[None, None, :])
        
        ray_vec = normalize(ray_vec)
        return ray_vec

